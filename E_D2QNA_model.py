import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import numpy as np
import random
import copy
import time
import json
import sys
from collections import deque
from torch_geometric.data import Data, Batch
from pathlib import Path
from sortedcontainers import SortedDict

# ==========================================
# Path Setup
# ==========================================
try:
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# Imports
# ==========================================
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix
# ⚡️⚡️⚡️ 关键修改：使用 PolicyNet1 (FiLM版) ⚡️⚡️⚡️
from networks.PolicyNet1 import Policy 
from networks.CreatDisjunctiveGraph import creatDisjunctiveGraph
from utils.data_loader import readData
from utils.scheduling_metrics import calculate_objectives
from utils.pareto import dedup_tolerant, non_dominated_filter_min
from utils.normalizer import SimpleNormalizer
from herisDispRules import eval as rule_eval
from utils.elite_archive import EliteArchive
from core.DynamicJobGenerator import DynamicJobGenerator
from envs.DynamicDJSSEnv import DynamicDJSSEnv

try:
    from torch_geometric.utils import dense_to_sparse
except Exception:
    def dense_to_sparse(adj):
        idx = torch.nonzero(adj > 0, as_tuple=False).t()
        vals = adj[idx[0], idx[1]]
        return idx, vals

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "model": {
        "raw_feat_dim": 11,   # 确保与环境输出一致
        "num_actions": 7,
        "embed_dim": 32,
        "gcn_hidden": 256,
        "num_heads": 8,
        "lstm_hidden": 128,
        "num_objectives": 2
    },
    "train": {
        "seed": 42,
        "lr": 5e-5,           
        "gamma": 0.99, 
        "batch_size": 32,     
        "memory_size": 5000,
        "episodes": 1000,
        "jobs_per_episode": 50,
        "save_interval": 50,
        "learn_every": 10,
        "num_alt_prefs": 2
    },
    "nsga2": {
        "population_size": 30, 
        "max_generations": 8, 
        "k_steps": 10,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2
    },
    "objectives_config": {
        "normalization": {
            "initial_ideal": [0.0, 0.0],
            "initial_nadir": [2000.0, 4000.0]
        }
    },
    "rules": ['SPT', 'LPT', 'SR', 'LR', 'FOPNR', 'MORPNR', 'RANDOM']
}

# ==========================================
# Multi-Fidelity Learner (The Brain)
# ==========================================
class MultiFidelityLearner:
    def __init__(self, policy, config, device, normalizer):
        self.policy = policy
        self.config = config
        self.device = device
        self.normalizer = normalizer
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=config["train"]["lr"],
            weight_decay=1e-4
        )
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.ce_loss_fn = torch.nn.CrossEntropyLoss() 
        self.gamma = config["train"]["gamma"]
        
        nsga = config["nsga2"]
        self.pop_size = nsga["population_size"]
        self.generations = nsga["max_generations"]
        self.k_steps = nsga["k_steps"]
        self.crossover_rate = nsga.get("crossover_rate", 0.8)
        self.mutation_rate = nsga.get("mutation_rate", 0.2)
        self.num_alt_prefs = config["train"]["num_alt_prefs"]
        self.num_actions = len(config["rules"])
        self.total_episodes = config["train"]["episodes"]
        
        self.ga_counter = 0
        self.archive_hits = 0
        self.elite_archive = EliteArchive(capacity=500)
        self.pf_points = []
        
        # 简单的精确匹配缓存 (防手抖重复计算)
        self.teacher_cache = {}
        
        # 记录每轮实际使用的预算
        self.teacher_budget_used = 0 

    def _get_dynamic_budget(self, current_ep):
        """RCATS 组件 1: 动态预算退火"""
        progress = current_ep / max(1, self.total_episodes)
        if progress < 0.15:
            return 150
        elif progress < 0.6:
            ratio = (progress - 0.15) / 0.45
            return int(150 - ratio * 100)
        else:
            return 50

    def _get_heuristic_confidence(self, current_ep):
        progress = current_ep / max(1, self.total_episodes)
        return 0.3 + 0.4 * min(1.0, progress)

    def _assess_criticality(self, snapshot):
        """关键度评估"""
        machines = snapshot['machines']
        jobs = snapshot['jobs']
        loads = [float(m['currentTime']) for m in machines]
        load_std = np.std(loads)
        avg_load = np.mean(loads) + 1e-6
        load_imbalance = np.clip(load_std / avg_load, 0, 1)
        
        max_time = max(loads)
        urgent_count = 0
        active_jobs = 0
        for j in jobs:
            uncompleted = [op for op in j['ops'] if not op['completed']]
            if not uncompleted: continue
            active_jobs += 1
            rem_work = sum(op['duration'] for op in uncompleted)
            slack = j['due'] - max_time - rem_work
            if slack < rem_work * 0.5: urgent_count += 1
        
        urgency = urgent_count / max(1, active_jobs)
        return 0.4 * load_imbalance + 0.6 * urgency

    def _select_teacher_samples(self, batch_memory, budget):
        """
        RCATS 组件 2: 分层主动采样 (Global Sorting)
        保留这个 V2 的核心逻辑，这是你的论文亮点！
        """
        n = len(batch_memory)
        if n == 0 or budget <= 0:
            return set()

        # 1. 计算所有样本的关键度
        scores = []
        for i, m in enumerate(batch_memory):
            if m['snapshot']:
                crit = self._assess_criticality(m['snapshot'])
            else:
                crit = 0.0
            scores.append((i, crit))
        
        # 按关键度降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 2. 分配预算 (80% Top-K, 20% Random)
        top_k_budget = int(budget * 0.8)
        random_budget = budget - top_k_budget
        
        selected_indices = set()
        
        # 选取 Top-K
        for i in range(min(top_k_budget, n)):
            idx, crit = scores[i]
            if crit > 0.3: # 门槛
                selected_indices.add(idx)
        
        # 选取 Random-M
        remaining_indices = [i for i in range(n) if i not in selected_indices]
        if remaining_indices and random_budget > 0:
            valid_remaining = [i for i in remaining_indices if batch_memory[i]['snapshot'] is not None]
            if valid_remaining:
                sampled = random.sample(valid_remaining, min(len(valid_remaining), random_budget))
                selected_indices.update(sampled)
                
        return selected_indices

    def _state_hash(self, snapshot, w):
        w_str = f"{w[0]:.2f}_{w[1]:.2f}"
        m_loads = sorted([int(m['currentTime']) for m in snapshot['machines']])
        m_str = "_".join(map(str, m_loads))
        n_jobs = len(snapshot['jobs'])
        rem_ops = sum(len([op for op in j['ops'] if not op['completed']]) for j in snapshot['jobs'])
        return f"{w_str}|{m_str}|{n_jobs}_{rem_ops}"

    def _create_state_snapshot(self, jobs, machines):
        return {
            'jobs': [
                {
                    'id': getattr(j, 'idItinerary', getattr(j, 'id', 0)),
                    'arrival': float(getattr(j, 'arrivalTime', 0.0)),
                    'due': float(getattr(j, 'due_date', float('inf'))),
                    'ops': [
                        {
                            'id': getattr(op, 'idOperation', 0),
                            'completed': bool(getattr(op, 'completed', False)),
                            'endTime': float(getattr(op, 'endTime', 0.0)),
                            'duration': float(getattr(op, 'duration', 0.0)),
                            'machine_opts': dict(getattr(op, 'machine', {})),
                            'assigned': getattr(op, 'assignedMachine', None)
                        }
                        for op in getattr(j, 'operations', [])
                    ]
                }
                for j in jobs
            ],
            'machines': [
                {
                    'name': getattr(m, 'name', ''),
                    'currentTime': float(getattr(m, 'currentTime', 0.0))
                }
                for m in machines
            ]
        }

    # ... _reconstruct_from_snapshot, _evaluate_chromosome_fast ...
    # (这部分代码通常很长且不变，此处略去以节省篇幅，请保持原样)
    def _reconstruct_from_snapshot(self, snapshot):
        # ... (保持原样) ...
        from core.clItinerary import Itinerary
        from core.cOperation import Operation
        from core.clMachine import Machine
        jobs = []
        for j_data in snapshot['jobs']:
            job = Itinerary(str(j_data['id']), int(j_data['due']) if j_data['due'] != float('inf') else 999999)
            job.arrivalTime = j_data['arrival']
            job.due_date = j_data['due']
            job.operations = []
            for op_data in j_data['ops']:
                op = Operation(parent_job=job, operation_id=int(op_data['id']), candidate_machines=dict(op_data['machine_opts']))
                op.completed = op_data['completed']
                op.endTime = op_data['endTime']
                op.duration = op_data['duration']
                op.assignedMachine = op_data['assigned']
                job.operations.append(op)
            jobs.append(job)
        machines = []
        for m_data in snapshot['machines']:
            machine = Machine(str(m_data['name']), float(m_data['currentTime']))
            machine.assignedOpera = [] 
            machines.append(machine)
        return jobs, machines

    def _evaluate_chromosome_fast(self, jobs, machines, chromosome, w_pref):
        # ... (保持原样) ...
        initial_state = []
        for j in jobs:
            for op in getattr(j, 'operations', []):
                initial_state.append((op, getattr(op, 'completed', False), getattr(op, 'endTime', 0.0), getattr(op, 'assignedMachine', None)))
        machine_times = {getattr(m, 'name', ''): float(getattr(m, 'currentTime', 0.0)) for m in machines}
        
        protime = SortedDict({0: 1})
        for action_idx in chromosome:
            rule = self.config["rules"][int(action_idx) % self.num_actions]
            try:
                _, jobs, protime = rule_eval(rule, jobs, machines, protime)
            except:
                break
            if all(all(op.completed for op in j.operations) for j in jobs): break
            
        current_time = max([m.currentTime for m in machines])
        avg_load = sum([m.currentTime for m in machines]) / len(machines)
        tt, ft = 0.0, 0.0
        for j in jobs:
            if not j.operations: continue
            if j.operations[-1].completed:
                comp = j.operations[-1].endTime
            else:
                rem = sum(op.duration for op in j.operations if not op.completed)
                comp = max(current_time, avg_load) + rem / len(machines) * 1.2
            ft += max(0, comp - j.arrivalTime)
            if j.due_date != 999999: tt += max(0, comp - j.due_date)
            
        raw = torch.tensor([tt, ft], dtype=torch.float32, device=self.device)
        h_val = self.normalizer.normalize(raw, to_negative=True)
        score = (h_val * w_pref).sum().item()
        obj_vec = [float(x) for x in h_val.tolist()]
        
        for (op, comp, end, mach) in initial_state:
            op.completed = comp; op.endTime = end; op.assignedMachine = mach
        for m in machines:
            m.currentTime = machine_times.get(m.name, 0.0); m.assignedOpera = []
        return score, obj_vec

    def _run_real_nsga2_k_step(self, snapshot, w_pref):
        jobs_base, machines_base = self._reconstruct_from_snapshot(snapshot)
        pop = np.random.randint(0, self.num_actions, (self.pop_size, self.k_steps))
        best_weighted_reward = -float('inf')
        best_obj_vec = None
        best_action = None
        
        elites = []
        
        for gen in range(self.generations):
            gen_rewards = []
            
            for indiv in pop:
                score, obj_vec = self._evaluate_chromosome_fast(jobs_base, machines_base, indiv, w_pref)
                gen_rewards.append(score)
                
                if score > best_weighted_reward:
                    best_weighted_reward = score
                    best_obj_vec = obj_vec
                    best_action = int(indiv[0])
            
            sorted_idx = np.argsort(gen_rewards)[::-1]
            elites = pop[sorted_idx[:self.pop_size//2]]
            
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = elites[random.randint(0, len(elites)-1)].copy()
                p2 = elites[random.randint(0, len(elites)-1)].copy()
                if random.random() < self.crossover_rate:
                    cut = random.randint(1, self.k_steps-1)
                    child = np.concatenate([p1[:cut], p2[cut:]])
                else:
                    child = p1
                
                mask = np.random.rand(self.k_steps) < self.mutation_rate
                if mask.any():
                    child[mask] = np.random.randint(0, self.num_actions, mask.sum())
                new_pop.append(child)
            pop = np.array(new_pop)
            
        return best_weighted_reward, best_obj_vec, best_action

    def _heuristic_value_covert_fast(self, snapshot):
        machines_data = snapshot['machines']
        jobs_data = snapshot['jobs']
        current_time = max(m['currentTime'] for m in machines_data)
        avg_machine_load = sum(m['currentTime'] for m in machines_data) / len(machines_data)
        tt, ft = 0.0, 0.0
        for j in jobs_data:
            uncompleted_ops = [op for op in j['ops'] if not op['completed']]
            if not uncompleted_ops: continue
            rem_work = sum(op['duration'] for op in uncompleted_ops)
            completion_time = max(current_time, avg_machine_load) + rem_work / max(1, len(machines_data)) * 1.2
            ft += max(0.0, completion_time - j['arrival'])
            if j['due'] != float('inf'): tt += max(0.0, completion_time - j['due'])
        raw = torch.tensor([tt, ft], dtype=torch.float32, device=self.device)
        return self.normalizer.normalize(raw, to_negative=True)

    def learn(self, memory, batch_size):
        if len(memory) < batch_size: return 0.0
        
        self.optimizer.zero_grad()
        
        current_ep = int(getattr(self, 'current_episode', 0))
        budget = self._get_dynamic_budget(current_ep)
        heuristic_conf_base = self._get_heuristic_confidence(current_ep)
        
        episodes = random.sample(memory, batch_size)
        seq_len = 10
        
        # 1. 收集所有 step，用于 RCATS 排序
        all_steps_flat = []
        processed_episodes = []
        
        for ep in episodes:
            if len(ep) <= seq_len:
                segment = ep
            else:
                start = random.randint(0, len(ep) - seq_len)
                segment = ep[start:start + seq_len]
            processed_episodes.append(segment)
            for m in segment:
                all_steps_flat.append(m)
        
        # 2. RCATS: 全局排序并分配 Budget
        teacher_indices = self._select_teacher_samples(all_steps_flat, budget)
        
        total_loss = 0.0
        total_steps = 0
        self.teacher_budget_used = len(teacher_indices)
        
        # 标记哪些样本用 Teacher
        for idx in range(len(all_steps_flat)):
            all_steps_flat[idx]['_use_teacher'] = (idx in teacher_indices)

        # 3. 按序列训练
        for segment in processed_episodes:
            h_state = self.policy.init_hidden_state(1)
            losses = []
            
            for m in segment:
                w = m['w'].to(self.device)
                feats = m['feats'].to(self.device)
                ei = m['ei'].to(self.device)
                r_vec = m['r_vec'].to(self.device)
                action_executed = m['action']
                
                use_teacher = False
                teacher_act = None
                target = 0.0
                confidence = 1.0
                
                if not m['done'] and m['snapshot'] is not None:
                    use_archive = False
                    archive_val = None
                    criticality = self._assess_criticality(m['snapshot'])

                    # 1) 先查表（Archive）
                    try:
                        if len(self.elite_archive.items) > 20:
                            w_np = w.detach().cpu().numpy()
                            similar_elites = []
                            for elite in self.elite_archive.items[-100:]:
                                elite_w = np.array(elite.get('pref', [0.0, 0.0]))
                                if np.linalg.norm(elite_w - w_np) < 0.3:
                                    if elite.get('obj', None) is not None:
                                        similar_elites.append(elite)
                            if similar_elites:
                                vals = []
                                for e in similar_elites:
                                    try:
                                        obj_vec = np.array(e.get('obj', [0.0, 0.0]), dtype=float)
                                        vals.append(float(np.dot(obj_vec, w_np)))
                                    except Exception:
                                        continue
                                if vals:
                                    archive_val = max(vals)
                    except Exception:
                        archive_val = None

                    # 2) 决策逻辑
                    if archive_val is not None:
                        target = archive_val
                        confidence = 1.0
                        use_archive = True
                        self.archive_hits += 1
                    elif m.get('_use_teacher', False) and (criticality > 0.5):
                        try:
                            state_key = self._state_hash(m['snapshot'], w)
                            if state_key in self.teacher_cache:
                                val, obj, act = self.teacher_cache[state_key]
                            else:
                                val, obj, act = self._run_real_nsga2_k_step(m['snapshot'], w)
                                self.teacher_cache[state_key] = (val, obj, act)
                            target = val
                            teacher_act = act
                            use_teacher = True
                            confidence = 1.0
                            self.ga_counter += 1
                            # 写入 Archive（保留精英）
                            try:
                                if obj is not None:
                                    self.elite_archive.add(pref=w.detach().cpu().numpy(), objectives=np.array(obj, dtype=float), traj=[], epoch=current_ep)
                            except Exception:
                                pass
                            # 仍然收集 PF 点用于可视化
                            if obj is not None:
                                try:
                                    self.pf_points.append({'episode': int(current_ep), 'obj': list(obj)})
                                except Exception:
                                    pass
                        except Exception:
                            use_teacher = False
                    else:
                        # 启发式退路
                        h_val = self._heuristic_value_covert_fast(m['snapshot'])
                        target = (h_val * w).sum().item()
                        confidence = heuristic_conf_base + 0.2 * (1.0 - float(criticality))
                
                elif m['done']:
                    target = (r_vec * w).sum().item()
                    confidence = 1.0

                # 传入 batch 索引（单图默认全零）并传递 w
                idx = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
                q_vecs, h_state = self.policy(feats, ei, h_state, batch=idx, w=w)
                
                # Loss Calculation
                train_act = teacher_act if (use_teacher and teacher_act is not None) else action_executed
                q_pred = (q_vecs[0, train_act] * w).sum()
                value_loss = (q_pred - target) ** 2 * confidence
                
                im_loss = torch.tensor(0.0, device=self.device)
                if use_teacher and teacher_act is not None:
                    q_logits = (q_vecs[0] * w.view(1, -1)).sum(dim=1).unsqueeze(0)
                    im_loss = self.ce_loss_fn(q_logits, torch.tensor([teacher_act], device=self.device))
                
                # Hybrid Loss
                loss = value_loss + 0.5 * im_loss
                losses.append(loss)
                total_steps += 1
                
                if '_use_teacher' in m: del m['_use_teacher']
            
            if losses:
                loss_sum = torch.stack(losses).sum()
                loss_sum.backward()
                total_loss += loss_sum.item()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss / max(1, total_steps)

# ==========================================
# Agent
# ==========================================
class FinalAgent:
    def __init__(self, policy, capacity, num_actions, device):
        self.policy = policy
        self.memory = deque(maxlen=int(capacity))
        self.current_episode_buffer = []
        self.num_actions = int(num_actions)
        self.device = device

    def remember(self, transition):
        self.current_episode_buffer.append(transition)
        if transition.get('done', False):
            self.memory.append(list(self.current_episode_buffer))
            self.current_episode_buffer = []

    def act(self, g, machines, hidden, w, epsilon):
        feats = get_features_of_node(g, machines, device=self.device)
        adj = get_adjacent_matrix(g, device=self.device)
        ei = adj.to_sparse().indices()
        
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                idx = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
                # ⚡️⚡️⚡️ 关键修正：传入 w 给 PolicyNet1 ⚡️⚡️⚡️
                w_input = w.view(1, -1) # [1, 2]
                q, hidden = self.policy(feats, ei, hidden, batch=idx, w=w_input)
                score = (q * w_input.view(1,1,-1)).sum(dim=2)
                action = score.argmax().item()
        return action, hidden, feats, ei

def sample_preference_curriculum(episode, total_episodes):
    progress = episode / max(1, total_episodes)
    if progress < 0.2:
        w = [1.0, 0.0] if random.random() < 0.5 else [0.0, 1.0]
    elif progress < 0.6:
        alpha = np.random.beta(0.5, 0.5, 2)
        w = alpha / alpha.sum()
    else:
        if random.random() < 0.35:
            w = [1.0, 0.0] if random.random() < 0.5 else [0.0, 1.0]
        else:
            alpha = np.random.dirichlet([1.0, 1.0])
            w = alpha
    return torch.tensor(w, dtype=torch.float32)

def evaluate_competitors(policy, jobs_tmpl, machines_tmpl, device, jobs_per_ep, episodes, w_eval):
    job_gen = DynamicJobGenerator(jobs_tmpl, config={"seed": 999, "urgent_prob": 0.3})
    env = DynamicDJSSEnv(job_gen, machines_tmpl, device)
    methods = ["SPT", "RANDOM", "DRL"]
    stats = {m: {"t": [], "f": []} for m in methods}
    for m in methods:
        env.job_gen.np_random.seed(999)
        for _ in range(episodes):
            g = env.reset(num_dynamic_jobs=jobs_per_ep)
            done = False
            hidden = None
            if m == "DRL":
                hidden = policy.init_hidden_state(1)
            while not done:
                if g is None:
                    g, _, done = env.step("RANDOM", multi_objective=True)
                    continue
                if m != "DRL":
                    rule = m
                else:
                    feats = get_features_of_node(g, env.machines, device=device)
                    adj = get_adjacent_matrix(g, device=device)
                    ei = adj.to_sparse().indices()
                    idx = torch.zeros(feats.size(0), dtype=torch.long, device=device)
                    need = int(getattr(policy.feat_embed, 'in_features', feats.shape[1]))
                    cur = int(feats.shape[1])
                    if cur != need:
                        w_row = w_eval.view(1, -1).repeat(feats.shape[0], 1)
                        if cur < need:
                            pad = w_row
                            if (cur + w_row.shape[1]) > need:
                                pad = w_row[:, : (need - cur)]
                            elif (cur + w_row.shape[1]) < need:
                                zeros = torch.zeros((feats.shape[0], need - cur - w_row.shape[1]), device=device)
                                pad = torch.cat([w_row, zeros], dim=1)
                            feats = torch.cat([feats, pad], dim=1)
                        else:
                            feats = feats[:, :need]
                    with torch.no_grad():
                        q, hidden = policy(feats, ei, hidden, batch=idx, w=w_eval)
                        score = (q * w_eval.view(1,1,-1)).sum(dim=2)
                        act = int(score.argmax().item())
                    rule = CONFIG["rules"][act]
                g, _, done = env.step(rule, multi_objective=True)
            stats[m]["t"].append(env.total_tardiness / jobs_per_ep)
            stats[m]["f"].append(getattr(env, 'total_flow_time', 0.0) / jobs_per_ep)
    return {
        "SPT": (float(np.mean(stats["SPT"]["t"])), float(np.mean(stats["SPT"]["f"]))),
        "RANDOM": (float(np.mean(stats["RANDOM"]["t"])), float(np.mean(stats["RANDOM"]["f"]))),
        "DRL": (float(np.mean(stats["DRL"]["t"])), float(np.mean(stats["DRL"]["f"]))),
    }

# ==========================================
# Main
# ==========================================
def main():
    seed = CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Running E-D2QNA FINAL V4 (Pure & Optimized) on {device}")
    
    # 路径管理
    json_path = "dataset/la16/la16_K1.3.json"
    if not os.path.exists(json_path): print("❌ Dataset not found"); return
    
    # 初始化
    _, jobs_tmpl, machines_tmpl = readData(json_path, validate=False)
    job_gen = DynamicJobGenerator(jobs_tmpl, config={"seed": seed, "urgent_prob": 0.3})
    env = DynamicDJSSEnv(job_gen, machines_tmpl, device)
    
    policy = Policy(CONFIG, device=str(device))
    normalizer = SimpleNormalizer(ideal=[0.,0.], nadir=[2000., 4000.])
    learner = MultiFidelityLearner(policy, CONFIG, device, normalizer)
    agent = FinalAgent(policy, CONFIG["train"]["memory_size"], len(CONFIG["rules"]), device)
    
    # 结果保存目录 (带时间戳)
    out_dir = Path("result/E_D2QNA_V4")
    session_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    session_dir = out_dir / session_ts
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "config.json", "w") as f: json.dump(str(CONFIG), f)
    
    log_file = session_dir / f"train_log_{session_ts}.csv"
    with open(log_file, "w") as f: f.write("episode,timestamp,avg_tard,avg_flow,loss,epsilon,time,ga_cnt,pf_cnt,tb_used\n")
    
    epsilon = 1.0
    # 移除了 Arc 列
    print(f"{'Ep':<6} {'Tardiness':<10} {'FlowTime':<10} {'Loss':<10} {'GA':<6} {'Eps':<6} {'PF':<6} {'TB':<6} {'Time':<6}")
    
    for ep in range(CONFIG["train"]["episodes"]):
        t0 = time.time()
        learner.ga_counter = 0
        learner.current_episode = ep + 1
        # 清空 PF 计数
        pf_cnt_start = len(learner.pf_points)
        
        g = env.reset(num_dynamic_jobs=CONFIG["train"]["jobs_per_episode"])
        hidden = policy.init_hidden_state(1)
        w = sample_preference_curriculum(ep, CONFIG["train"]["episodes"]).to(device)
        
        done = False
        steps = 0
        total_loss = 0
        loss_cnt = 0
        tb_sum = 0
        tb_cnt = 0
        
        while not done:
            if g is None:
                g, _, done = env.step("RANDOM", multi_objective=True)
                continue
            
            action, hidden, feats, ei = agent.act(g, env.machines, hidden, w, epsilon)
            rule = CONFIG["rules"][action]
            
            snapshot = learner._create_state_snapshot(env.active_jobs, env.machines)
            g_next, r_vec, done = env.step(rule, multi_objective=True)
            
            r_vec_t = torch.tensor(r_vec, dtype=torch.float32)
            agent.remember({
                'feats': feats.cpu(), 'ei': ei.cpu(),
                'action': action, 'r_vec': r_vec_t, 
                'done': done, 'w': w.cpu(),
                'snapshot': snapshot 
            })
            
            if len(agent.memory) >= CONFIG["train"]["batch_size"] and steps % CONFIG["train"]["learn_every"] == 0:
                l = learner.learn(agent.memory, CONFIG["train"]["batch_size"])
                total_loss += l
                loss_cnt += 1
                tb_sum += getattr(learner, 'teacher_budget_used', 0)
                tb_cnt += 1
            
            g = g_next
            steps += 1
            
        if len(agent.memory) > 0:
            bs = min(CONFIG["train"]["batch_size"], len(agent.memory))
            l = learner.learn(agent.memory, bs)
            total_loss += l
            loss_cnt += 1
            tb_sum += getattr(learner, 'teacher_budget_used', 0)
            tb_cnt += 1

        epsilon = max(0.05, epsilon * 0.995)
        avg_loss = total_loss / max(1, loss_cnt)
        n_jobs = CONFIG["train"]["jobs_per_episode"]
        
        pf_cnt_ep = len(learner.pf_points) - pf_cnt_start
        tb_avg = tb_sum / max(1, tb_cnt)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{ep+1:<6} {env.total_tardiness/n_jobs:<10.1f} {env.total_flow_time/n_jobs:<10.1f} {avg_loss:<10.2f} {learner.ga_counter:<6} {epsilon:<6.2f} {pf_cnt_ep:<6} {tb_avg:<6.1f} {time.time()-t0:<6.1f}")
        
        if (ep+1) % 10 == 0:
            ts_print = time.strftime("%H:%M:%S")
            print(f"=== Checkpoint Ep {ep+1} @ {ts_print} ===")
        
        with open(log_file, "a") as f:
            f.write(f"{ep+1},{ts},{env.total_tardiness/n_jobs},{env.total_flow_time/n_jobs},{avg_loss},{epsilon},{time.time()-t0},{learner.ga_counter},{pf_cnt_ep},{tb_avg}\n")
            
        if (ep+1) % 100 == 0:
            ckpt_path = session_dir / f"ep_{ep+1}.pth"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"*** Model checkpoint saved: {ckpt_path}")
            bench_path = session_dir / "benchmark.csv"
            w_eval = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
            res = evaluate_competitors(policy, jobs_tmpl, machines_tmpl, device, CONFIG["train"]["jobs_per_episode"], 50, w_eval)
            header = "Model,SPT Avg Tardiness,SPT Mean Flow Time,RANDOM Avg Tardiness,RANDOM Mean Flow Time,DRL_Agent Avg Tardiness,DRL_Agent Mean Flow Time\n"
            row = f"ep_{ep+1}.pth,{res['SPT'][0]},{res['SPT'][1]},{res['RANDOM'][0]},{res['RANDOM'][1]},{res['DRL'][0]},{res['DRL'][1]}\n"
            if not bench_path.exists():
                with open(bench_path, "w") as bf:
                    bf.write(header)
            with open(bench_path, "a") as bf:
                bf.write(row)

if __name__ == "__main__":
    main()
