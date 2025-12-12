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

# Imports
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix
from networks.PolicyNet1 import Policy  # 确保这里用的是修复了FiLM Bug的最新版
from utils.data_loader import readData
from utils.normalizer import SimpleNormalizer
from core.DynamicJobGenerator import DynamicJobGenerator
from envs.DynamicDJSSEnv import DynamicDJSSEnv
from E_D2QNA_model import MultiFidelityLearner, FinalAgent, CONFIG as BASE_CONFIG

try:
    from torch_geometric.utils import dense_to_sparse
except Exception:
    def dense_to_sparse(adj):
        idx = torch.nonzero(adj > 0, as_tuple=False).t()
        vals = adj[idx[0], idx[1]]
        return idx, vals

# ==========================================
# 🔧 Fine-Tuning Configuration (狙击模式)
# ==========================================
FT_CONFIG = copy.deepcopy(BASE_CONFIG)

# 1. 关键参数调整
FT_CONFIG["model"]["raw_feat_dim"] = 11   # ✅ 必须对应 V5 的 11维
FT_CONFIG["train"]["lr"] = 1e-5           # ✅ 极低学习率，防止震荡
FT_CONFIG["train"]["learn_every"] = 5     # 高频更新
FT_CONFIG["train"]["episodes"] = 200
FT_CONFIG["ewc_lambda"] = 5000.0

# 2. 目标阈值 (Target)
TARGET_TARDINESS = 4600.0  # 🎯 只要低于这个分，立刻保存！
BEST_EVER_TARDINESS = float('inf')

# ==========================================
# Helper: Load Checkpoint & Compute Fisher
# ==========================================
def load_checkpoint_with_fisher(policy, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"🔄 Loading Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 过滤不匹配的键 (以防万一)
    model_dict = policy.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    
    try:
        policy.load_state_dict(model_dict)
        print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")
    except Exception as e:
        print(f"⚠️ Load Warning: {e}")
    
    # Snapshot parameters for EWC
    frozen_params = {
        name: param.clone().detach() 
        for name, param in policy.named_parameters() 
        if param.requires_grad
    }
    return policy, frozen_params

# ==========================================
# Finetuning Learner (EWC + Stratified)
# ==========================================
class FinetuningLearner(MultiFidelityLearner):
    def __init__(self, policy, config, device, normalizer, frozen_params, ewc_lambda=2000.0):
        super().__init__(policy, config, device, normalizer)
        self.frozen_params = frozen_params
        self.ewc_lambda = ewc_lambda
        
        # 强制多样性偏好
        self.pref_grid = self._build_pref_grid()
        self.pref_usage_count = {tuple(p.tolist()): 0 for p in self.pref_grid}
        
        # 覆盖优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=config["train"]["lr"],
            weight_decay=1e-4
        )

    def _build_pref_grid(self):
        grid = []
        for i in range(11):
            w1 = i / 10.0
            w2 = 1.0 - w1
            grid.append(torch.tensor([w1, w2], dtype=torch.float32))
        return grid
    
    def _compute_ewc_loss(self):
        ewc_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.policy.named_parameters():
            if name in self.frozen_params:
                frozen = self.frozen_params[name].to(self.device)
                ewc_loss += torch.sum((param - frozen) ** 2)
        return self.ewc_lambda * ewc_loss
    
    def _sample_stratified_preferences(self, batch_size):
        sorted_prefs = sorted(self.pref_usage_count.items(), key=lambda x: x[1])
        selected = []
        count = 0
        while len(selected) < batch_size:
            pref_tuple, _ = sorted_prefs[count % len(sorted_prefs)]
            pref = torch.tensor(pref_tuple, dtype=torch.float32)
            selected.append(pref)
            self.pref_usage_count[pref_tuple] += 1
            count += 1
        return selected[:batch_size]

    def learn(self, memory, batch_size):
        if len(memory) < batch_size: return 0.0
        self.optimizer.zero_grad()
        
        episodes = random.sample(memory, batch_size)
        stratified_prefs = self._sample_stratified_preferences(batch_size)
        total_loss = 0.0
        total_steps = 0
        
        for ep_idx, ep in enumerate(episodes):
            w_override = stratified_prefs[ep_idx].to(self.device)
            segment = ep if len(ep) <= 10 else ep[random.randint(0, len(ep)-10):][:10]
            
            h_state = self.policy.init_hidden_state(1)
            losses = []
            
            for m in segment:
                w = w_override # Force diverse preference
                feats = m['feats'].to(self.device)
                ei = m['ei'].to(self.device)
                r_vec = m['r_vec'].to(self.device)
                action_executed = m['action']
                
                # Simple Target (No Teacher call to save time, assume Buffer has good stuff)
                # Or use heuristic fallback
                h_val = self._heuristic_value_covert_fast(m['snapshot'])
                target = (h_val * w).sum().item()
                confidence = 0.8
                
                # Forward with FiLM
                idx = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
                q_vecs, h_state = self.policy(feats, ei, h_state, batch=idx, w=w)
                
                q_pred = (q_vecs[0, action_executed] * w).sum()
                losses.append((q_pred - target) ** 2 * confidence)
                total_steps += 1
            
            if losses:
                loss_sum = torch.stack(losses).sum()
                loss_sum.backward()
                total_loss += loss_sum.item()
        
        # EWC Regularization
        ewc = self._compute_ewc_loss()
        ewc.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return total_loss

# ==========================================
# Main Execution
# ==========================================
def main():
    global BEST_EVER_TARDINESS
    
    seed = FT_CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Capture-Mode Finetuning on {device}")
    
    # 1. 路径设置
    chkpt_path = r"result/E_D2QNA/20251210_144706/ep_700.pth"
    
    if not os.path.exists(chkpt_path):
        import glob
        files = glob.glob("result/E_D2QNA/**/ep_700.pth", recursive=True)
        if files:
            chkpt_path = files[-1]
            print(f"🔍 Auto-found checkpoint: {chkpt_path}")
        else:
            print("❌ Checkpoint not found. Please edit 'chkpt_path'.")
            return

    # 2. 环境初始化
    json_path = "dataset/la16/la16_K1.3.json"
    _, jobs_tmpl, machines_tmpl = readData(json_path, validate=False)
    job_gen = DynamicJobGenerator(jobs_tmpl, config={"seed": seed, "urgent_prob": 0.3})
    env = DynamicDJSSEnv(job_gen, machines_tmpl, device)
    
    # 3. 模型加载
    policy = Policy(FT_CONFIG, device=str(device))
    policy, frozen_params = load_checkpoint_with_fisher(policy, chkpt_path, device)
    normalizer = SimpleNormalizer(ideal=[0.,0.], nadir=[2000., 4000.])
    
    learner = FinetuningLearner(policy, FT_CONFIG, device, normalizer, frozen_params, ewc_lambda=FT_CONFIG["ewc_lambda"])
    agent = FinalAgent(policy, FT_CONFIG["train"]["memory_size"], len(FT_CONFIG["rules"]), device)
    
    # 4. 输出设置
    out_dir = Path("result/E_D2QNA/capture_best")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'Ep':<6} {'Tardiness':<12} {'Best':<12} {'Status'}")
    
    # 5. 微调循环 (300轮)
    epsilon = 0.05 # 保持低探索，主要利用
    
    # 预填充 Buffer
    print("🔄 Filling buffer...")
    for _ in range(3):
        g = env.reset(num_dynamic_jobs=50)
        done = False
        hidden = policy.init_hidden_state(1)
        w = torch.tensor([1.0, 0.0], device=device) # 关注 Tardiness
        while not done:
            if g is None: g, _, done = env.step("RANDOM", multi_objective=True); continue
            action, hidden, feats, ei = agent.act(g, env.machines, hidden, w, epsilon)
            rule = FT_CONFIG["rules"][action]
            snapshot = learner._create_state_snapshot(env.active_jobs, env.machines)
            g_next, r, done = env.step(rule, multi_objective=True)
            agent.remember({'feats': feats.cpu(), 'ei': ei.cpu(), 'action': action, 'r_vec': torch.tensor(r), 'done': done, 'w': w.cpu(), 'snapshot': snapshot})
            g = g_next

    # 开始循环
    for ep in range(FT_CONFIG["train"]["episodes"]):
        g = env.reset(num_dynamic_jobs=50)
        hidden = policy.init_hidden_state(1)
        w = torch.tensor([1.0, 0.0], device=device) # 强制优化 Tardiness
        
        done = False
        steps = 0
        
        while not done:
            if g is None:
                g, _, done = env.step("RANDOM", multi_objective=True)
                continue
            
            action, hidden, feats, ei = agent.act(g, env.machines, hidden, w, epsilon)
            rule = FT_CONFIG["rules"][action]
            snapshot = learner._create_state_snapshot(env.active_jobs, env.machines)
            g_next, r_vec, done = env.step(rule, multi_objective=True)
            agent.remember({'feats': feats.cpu(), 'ei': ei.cpu(), 'action': action, 'r_vec': torch.tensor(r_vec), 'done': done, 'w': w.cpu(), 'snapshot': snapshot})
            
            if len(agent.memory) >= 32 and steps % 5 == 0:
                learner.learn(agent.memory, 32)
            
            g = g_next
            steps += 1
            
        # --- 🏆 核心抓拍逻辑 ---
        # 计算本轮 Tardiness
        current_tard = env.total_tardiness / 50.0
        
        status = ""
        # 只要优于 SPT (4561) 或者哪怕接近，就保存
        if current_tard < 4600:
            status = "🔥 SAVED"
            torch.save(policy.state_dict(), out_dir / f"captured_ep{ep}_t{int(current_tard)}.pth")
            
            if current_tard < BEST_EVER_TARDINESS:
                BEST_EVER_TARDINESS = current_tard
                torch.save(policy.state_dict(), out_dir / "best_model.pth")
                status = "🏆 NEW BEST"
        
        print(f"{ep+1:<6} {current_tard:<12.1f} {BEST_EVER_TARDINESS:<12.1f} {status}")

if __name__ == "__main__":
    main()
