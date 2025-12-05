import torch
import numpy as np
import os
import random
import copy
import time
from collections import deque
from torch_geometric.data import Data, Batch
from pathlib import Path

# 导入你的模块
from networks.PolicyNet import Policy
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix
from core.DynamicJobGenerator import DynamicJobGenerator
from envs.DynamicDJSSEnv import DynamicDJSSEnv
from utils.data_loader import readData
from utils.normalizer import SimpleNormalizer
os.environ["DEBUG_ASSERT"] = "0"
# --- 配置参数 ---
CONFIG = {
    "model": {
        "raw_feat_dim": 11,   # 基础特征维度
        "num_actions": 7,     # 规则数量
        "embed_dim": 32,
        "gcn_hidden": 64,
        "num_heads": 4,
        "lstm_hidden": 64,
        "num_objectives": 2   
    },
    "train": {
        "seed": 42,
        "lr": 1e-4, 
        "gamma": 0.99, 
        "batch_size": 32,
        "memory_size": 5000,
        "episodes": 1000,
        "jobs_per_episode": 50,
        "save_interval": 100,
        "learn_every": 10     
    },
    "rules": ['SPT', 'LPT', 'SR', 'LR', 'FOPNR', 'MORPNR', 'RANDOM'],
    "objectives_config": {
        "active_objectives": ["tardiness", "flow_time"],
        "normalization": {
            "enabled": True,
            "initial_ideal": [0.0, 0.0],
            "initial_nadir": [1000.0, 2000.0] 
        }
    }
}

class StandardMODQNTrainer:
    def __init__(self, policy, optimizer, device, normalizer, target_update_freq=1000):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.normalizer = normalizer
        
        # ✅ 修正1: 引入 Target Network
        self.target_policy = copy.deepcopy(policy)
        self.target_policy.eval() # Target Net 不参与反向传播
        
        self.memory = deque(maxlen=CONFIG["train"]["memory_size"])
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = CONFIG["train"]["gamma"]
        
        self.target_update_freq = target_update_freq
        self.update_cnt = 0

    def select_action(self, feats, edge_index, hidden, epsilon, w):
        if random.random() < epsilon:
            return random.randint(0, CONFIG["model"]["num_actions"] - 1), hidden

        batch = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            q_values, new_hidden = self.policy(feats, edge_index, hidden, batch=batch)
            w_tensor = w.view(1, 1, -1).to(self.device)
            # Scalarization: Q_scalar = sum(Q_vec * w)
            q_scalar = (q_values * w_tensor).sum(dim=2)
            action = q_scalar.argmax(dim=1).item()
        return action, new_hidden

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < CONFIG["train"]["batch_size"]:
            return 0.0
            
        batch_data = random.sample(self.memory, CONFIG["train"]["batch_size"])
        
        q_evals = []
        q_targets = []
        
        # 批量处理通常更高效，这里为了逻辑清晰保持循环，也可以优化
        for item in batch_data:
            f, ei, a, r_vec, f_next, ei_next, d, w = item
            w_dev = w.to(self.device)
            r_vec_dev = r_vec.to(self.device)

            # --- 1. Current Q (用 Policy Net) ---
            data = Data(x=f.to(self.device), edge_index=ei.to(self.device))
            batch = Batch.from_data_list([data]).to(self.device)
            h0 = self.policy.init_hidden_state(1)
            q_out, _ = self.policy(batch.x, batch.edge_index, h0, batch=batch.batch)
            q_scalar = (q_out[0, a] * w_dev).sum()
            q_evals.append(q_scalar)

            # --- 2. Target Q (用 Target Net) ---
            # ✅ 修正: 计算 Target 必须用 Target Network
            if not d and f_next is not None:
                data_next = Data(x=f_next.to(self.device), edge_index=ei_next.to(self.device))
                batch_next = Batch.from_data_list([data_next]).to(self.device)
                h0_next = self.target_policy.init_hidden_state(1) # 注意这里是 target_policy
                
                with torch.no_grad():
                    # Double DQN 技巧 (可选，推荐加上，显得更专业)
                    # 1. 用 Policy Net 选动作
                    q_next_online, _ = self.policy(batch_next.x, batch_next.edge_index, h0_next, batch=batch_next.batch)
                    q_next_online_scalar = (q_next_online * w_dev.view(1, 1, -1)).sum(dim=2)
                    best_action = q_next_online_scalar.argmax(dim=1).item()
                    
                    # 2. 用 Target Net 算价值
                    q_next_target, _ = self.target_policy(batch_next.x, batch_next.edge_index, h0_next, batch=batch_next.batch)
                    q_val = (q_next_target[0, best_action] * w_dev).sum().item()
                    
                    target = (r_vec_dev * w_dev).sum().item() + self.gamma * q_val
            else:
                target = (r_vec_dev * w_dev).sum().item()
            
            q_targets.append(target)
        
        q_preds_tensor = torch.stack(q_evals)
        q_targets_tensor = torch.tensor(q_targets, device=self.device)

        loss = self.loss_fn(q_preds_tensor, q_targets_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # ✅ 修正: 定期更新 Target Network
        self.update_cnt += 1
        if self.update_cnt % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            # print("Target Network Updated")
        
        return loss.item()

def main():
    seed = CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Running Focused DQN (Tardiness Killer) on {device}")
    
    json_path = r"dataset/la16/la16_K1.3.json"
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    _, jobs, machines = readData(json_path, validate=False)
    
    # ⚡️⚡️⚡️ 关键修改 1: 开启困难模式 (Trap Mode) ⚡️⚡️⚡️
    # urgent_prob = 0.3 (30% 概率出现紧急长任务，SPT 必死)
    job_gen = DynamicJobGenerator(jobs, config={"seed": seed, "urgent_prob": 0.3})
    
    env = DynamicDJSSEnv(job_gen, machines, device)
    
    policy = Policy(CONFIG, device=str(device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=CONFIG["train"]["lr"])
    
    ncfg = CONFIG["objectives_config"]["normalization"]
    normalizer = SimpleNormalizer(ideal=ncfg["initial_ideal"], nadir=ncfg["initial_nadir"])
    
    trainer = StandardMODQNTrainer(policy, optimizer, device, normalizer, target_update_freq=500)
    
    repo_root = Path(__file__).resolve().parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "result" / "DQN" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({"config": CONFIG}, out_dir / "config.pth")

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("episode,reward,steps,loss,epsilon,avg_tard,avg_flow\n")

    start_episode = 0
    target_episodes = CONFIG["train"]["episodes"]
    epsilon = 1.0
    
    print("Start Training...")

    for episode in range(target_episodes):
        g = env.reset(num_dynamic_jobs=CONFIG["train"]["jobs_per_episode"])
        env.completed_jobs = []
        hidden = policy.init_hidden_state(batch_size=1)
        
        if random.random() < 0.5:
            w = torch.tensor([1.0, 0.0], dtype=torch.float32)
        else:
            alpha = np.random.dirichlet([1.0, 1.0])
            w = torch.tensor(alpha, dtype=torch.float32)
        
        total_reward_scalar = 0
        done = False
        steps = 0
        loss = 0
        
        while not done:
            if g is None:
                g_next, r, done = env.step("RANDOM", multi_objective=True)
                g = g_next
                continue

            feats = get_features_of_node(g, env.machines, device=device)
            adj = get_adjacent_matrix(g, device=device)
            edge_index = adj.to_sparse().indices()
            
            action, hidden = trainer.select_action(feats, edge_index, hidden, epsilon, w)
            rule_name = CONFIG["rules"][action]
            
            g_next, reward_vec, done = env.step(rule_name, multi_objective=True)
            
            r_vec_tensor = torch.tensor(reward_vec, dtype=torch.float32)

            if g_next is None:
                feats_next = None
                edge_index_next = None
            else:
                feats_next = get_features_of_node(g_next, env.machines, device=device)
                adj_next = get_adjacent_matrix(g_next, device=device)
                edge_index_next = adj_next.to_sparse().indices()

            trainer.store((
                feats.detach().cpu(), edge_index.detach().cpu(), 
                action, r_vec_tensor,
                None if feats_next is None else feats_next.detach().cpu(),
                None if edge_index_next is None else edge_index_next.detach().cpu(),
                bool(done),
                w # 存入固定的 w
            ))
            
            if steps % CONFIG["train"]["learn_every"] == 0:
                loss = trainer.update()
            
            total_reward_scalar += (r_vec_tensor * w).sum().item()
            g = g_next
            steps += 1
            
        epsilon = max(0.05, epsilon * 0.99)
        
        tardiness = env.total_tardiness
        flow_time = getattr(env, 'total_flow_time', 0.0)

        global_ep = start_episode + episode + 1
        n_jobs = CONFIG["train"]["jobs_per_episode"]
        avg_tard = tardiness / n_jobs
        avg_flow = flow_time / n_jobs

        print(f"Episode {global_ep}/{target_episodes}: "
            f"Rw={total_reward_scalar:.2f}, Loss={loss:.4f}, Eps={epsilon:.2f}, "
            f"AvgTard={avg_tard:.2f}, AvgFlow={avg_flow:.2f}") 
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{global_ep},{total_reward_scalar:.2f},{steps},{loss:.4f},{epsilon:.2f},{avg_tard},{avg_flow}\n")
        
        if (global_ep) % CONFIG["train"]["save_interval"] == 0:
            ckpt_path = out_dir / f"epoch_{global_ep:04d}.pth"
            torch.save({
                'model_state_dict': policy.state_dict(),
                'config': CONFIG,
                'epoch': global_ep
            }, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
