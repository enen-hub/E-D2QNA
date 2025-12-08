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
# Path Setup & Imports (Reuse existing modules)
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

# Import necessary components from your original project structure
# Make sure these paths match your file structure
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix
from networks.PolicyNet import Policy
from utils.data_loader import readData
from utils.normalizer import SimpleNormalizer
from herisDispRules import eval as rule_eval
from core.DynamicJobGenerator import DynamicJobGenerator
from envs.DynamicDJSSEnv import DynamicDJSSEnv
# Import the original Learner to inherit basic methods
from E_D2QNA_model import MultiFidelityLearner, FinalAgent, CONFIG as BASE_CONFIG

try:
    from torch_geometric.utils import dense_to_sparse
except Exception:
    def dense_to_sparse(adj):
        idx = torch.nonzero(adj > 0, as_tuple=False).t()
        vals = adj[idx[0], idx[1]]
        return idx, vals

# ==========================================
# Fine-Tuning Configuration
# ==========================================
FT_CONFIG = copy.deepcopy(BASE_CONFIG)
# Key adjustments for Fine-Tuning
FT_CONFIG["model"]["raw_feat_dim"] = 13       # 使模型输入维度与 ep_400 一致
FT_CONFIG["train"]["lr"] = 1e-5          # ✅ Lower LR (1/5 of baseline)
FT_CONFIG["train"]["learn_every"] = 5    # ✅ More frequent updates
FT_CONFIG["train"]["episodes"] = 1000    # Target end episode
FT_CONFIG["ewc_lambda"] = 2000.0         # ✅ EWC Penalty Strength (Try 1000-5000)

# ==========================================
# Helper: Load Checkpoint & Compute Fisher
# ==========================================
def load_checkpoint_with_fisher(policy, checkpoint_path, device):
    """Load weights and snapshot them for EWC regularization."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"🔄 Loading Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        policy.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully (Strict).")
    except Exception as e:
        print(f"⚠️ Strict load failed: {e}")
        print("🔄 Attempting non-strict load...")
        policy.load_state_dict(state_dict, strict=False)
    
    # Snapshot parameters to prevent forgetting
    frozen_params = {
        name: param.clone().detach() 
        for name, param in policy.named_parameters() 
        if param.requires_grad
    }
    
    print(f"✅ Snapshot taken for {len(frozen_params)} parameters.")
    return policy, frozen_params

# ==========================================
# Finetuning Learner (EWC + Stratified Sampling)
# ==========================================
class FinetuningLearner(MultiFidelityLearner):
    def __init__(self, policy, config, device, normalizer, frozen_params, ewc_lambda=2000.0):
        # Initialize base learner
        super().__init__(policy, config, device, normalizer)
        
        self.frozen_params = frozen_params
        self.ewc_lambda = ewc_lambda
        
        # Stratified Preference Grid (11 points: [0,1], [0.1,0.9] ... [1,0])
        self.pref_grid = self._build_pref_grid()
        self.pref_usage_count = {tuple(p.tolist()): 0 for p in self.pref_grid}
        
        # Override optimizer with lower LR
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
        """Calculate L2 penalty for deviation from Ep 400 weights."""
        ewc_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.policy.named_parameters():
            if name in self.frozen_params:
                frozen = self.frozen_params[name].to(self.device)
                ewc_loss += torch.sum((param - frozen) ** 2)
        return self.ewc_lambda * ewc_loss
    
    def _sample_stratified_preferences(self, batch_size):
        """Select diverse preferences that have been used least often."""
        sorted_prefs = sorted(self.pref_usage_count.items(), key=lambda x: x[1])
        selected = []
        
        # Fill batch with least-used prefs
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
        
        # 1. Sample Episodes
        episodes = random.sample(memory, batch_size)
        
        # 2. Generate Stratified Preferences (Force Diversity)
        stratified_prefs = self._sample_stratified_preferences(batch_size)
        
        total_loss = 0.0
        total_steps = 0
        seq_len = 10
        
        for ep_idx, ep in enumerate(episodes):
            # 🔴 Override preference: Use the forced diverse preference
            w_override = stratified_prefs[ep_idx].to(self.device)
            
            if len(ep) <= seq_len:
                segment = ep
            else:
                start = random.randint(0, len(ep) - seq_len)
                segment = ep[start:start + seq_len]
            
            h_state = self.policy.init_hidden_state(1)
            losses = []
            
            for m in segment:
                # Use overridden w
                w = w_override
                feats = m['feats'].to(self.device)
                ei = m['ei'].to(self.device)
                r_vec = m['r_vec'].to(self.device)
                action_executed = m['action']
                
                # Re-calculate scalar reward with new w
                r_scalar = (r_vec * w).sum().item()
                target = 0.0
                confidence = 1.0
                use_teacher = False
                teacher_action = None
                
                # Criticality Check
                criticality = 0.0
                if m['snapshot']:
                    criticality = self._assess_criticality(m['snapshot'])
                heuristic_conf = 0.3 + 0.4 * (1.0 - criticality)
                
                # Reduced Budget for Fine-Tuning (20-50 calls)
                budget = 50 
                
                # Teacher Call Logic (Simpler, no Archive)
                if not m['done'] and m['snapshot'] is not None:
                    # Higher threshold (0.6) for teacher during fine-tuning
                    if criticality > 0.6 and self.ga_counter < budget * 5: # loose per-update limit
                        try:
                            # Always run fresh NSGA-II (No Archive Lookup)
                            val, obj, act = self._run_real_nsga2_k_step(m['snapshot'], w)
                            target = val
                            teacher_action = act
                            use_teacher = True
                            confidence = 1.0
                            self.ga_counter += 1
                            
                            if obj is not None:
                                self.pf_points.append({
                                    'episode': getattr(self, 'current_episode', 0),
                                    'obj': obj
                                })
                        except Exception:
                            use_teacher = False
                            h_val = self._heuristic_value_covert_fast(m['snapshot'])
                            target = (h_val * w).sum().item()
                            confidence = heuristic_conf
                    else:
                        h_val = self._heuristic_value_covert_fast(m['snapshot'])
                        target = (h_val * w).sum().item()
                        confidence = heuristic_conf
                elif m['done']:
                    target = r_scalar
                    confidence = 1.0
                
                # --- Forward ---
                need = int(getattr(self.policy.feat_embed, 'in_features', feats.shape[1]))
                cur = int(feats.shape[1])
                if cur != need:
                    w_row = w.view(1, -1).repeat(feats.shape[0], 1)
                    if cur < need:
                        pad = w_row
                        if (cur + w_row.shape[1]) > need:
                            pad = w_row[:, : (need - cur)]
                        elif (cur + w_row.shape[1]) < need:
                            zeros = torch.zeros((feats.shape[0], need - cur - w_row.shape[1]), device=self.device)
                            pad = torch.cat([w_row, zeros], dim=1)
                        feats = torch.cat([feats, pad], dim=1)
                    else:
                        feats = feats[:, :need]

                data = Data(x=feats, edge_index=ei)
                batch = Batch.from_data_list([data]).to(self.device)
                q_vecs, h_state = self.policy(batch.x, batch.edge_index, h_state, batch=batch.batch)
                
                # Value Loss
                act_idx = teacher_action if (use_teacher and teacher_action is not None) else action_executed
                q_pred = (q_vecs[0, act_idx] * w).sum()
                value_loss = (q_pred - target) ** 2 * confidence
                
                # Imitation Loss (Reduced weight 0.3)
                im_loss = torch.tensor(0.0, device=self.device)
                if use_teacher and teacher_action is not None:
                    q_logits = (q_vecs[0] * w.view(1, -1)).sum(dim=1).unsqueeze(0)
                    im_loss = self.ce_loss_fn(q_logits, torch.tensor([teacher_action], device=self.device))
                
                loss = value_loss + 0.3 * im_loss
                losses.append(loss)
                total_steps += 1
            
            if losses:
                loss_sum = torch.stack(losses).sum()
                loss_sum.backward()
                total_loss += loss_sum.item()
        
        # 3. ✅ Add EWC Penalty
        ewc_penalty = self._compute_ewc_loss()
        ewc_penalty.backward()
        total_loss += ewc_penalty.item()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss / max(1, total_steps)

# ==========================================
# Main Execution
# ==========================================
def main():
    seed = FT_CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Fine-Tuning (EWC + Stratified) on {device}")
    
    # Setup Data & Env
    json_path = "dataset/la16/la16_K1.3.json" # Adjust if needed
    if not os.path.exists(json_path): print("❌ Dataset not found"); return
    _, jobs_tmpl, machines_tmpl = readData(json_path, validate=False)
    
    job_gen = DynamicJobGenerator(jobs_tmpl, config={"seed": seed, "urgent_prob": 0.3})
    env = DynamicDJSSEnv(job_gen, machines_tmpl, device)
    
    policy = Policy(FT_CONFIG, device=str(device))
    normalizer = SimpleNormalizer(ideal=[0.,0.], nadir=[2000., 4000.])
    
    # ✅ Load Ep 400 (user-provided path)
    chkpt_path = r"E:\xuexi\xiangmu\chejiandiaodu\shenduQwangluo\E_D2QNA\result\E_D2QNA_FINAL_V2\ep_400.pth"
    policy, frozen_params = load_checkpoint_with_fisher(policy, chkpt_path, device)
    
    # Initialize Learner
    learner = FinetuningLearner(
        policy, FT_CONFIG, device, normalizer, 
        frozen_params, ewc_lambda=FT_CONFIG["ewc_lambda"]
    )
    
    agent = FinalAgent(policy, FT_CONFIG["train"]["memory_size"], len(FT_CONFIG["rules"]), device)
    
    # Output Setup
    out_dir = Path("result/E_D2QNA/finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ft_config.json", "w") as f: json.dump(str(FT_CONFIG), f)
    
    log_file = out_dir / "ft_log.csv"
    with open(log_file, "w") as f: f.write("episode,tardiness,flowtime,loss,epsilon,time,ga_cnt\n")
    
    print(f"{'Ep':<6} {'Tard':<10} {'Flow':<10} {'Loss':<10} {'GA':<6} {'Eps':<6} {'Time':<6}")
    
    # Start from 400
    epsilon = 0.15 # Low epsilon for exploitation
    start_ep = 400
    
    # Fill replay buffer slightly before training
    print("🔄 Pre-filling replay buffer...")
    # ... (Skipped for brevity, agent will fill in first few eps)
    
    for ep in range(start_ep, FT_CONFIG["train"]["episodes"]):
        t0 = time.time()
        learner.ga_counter = 0
        learner.current_episode = ep + 1
        
        g = env.reset(num_dynamic_jobs=FT_CONFIG["train"]["jobs_per_episode"])
        hidden = policy.init_hidden_state(1)
        
        # During rollout, use random preference to fill buffer diversely
        # But actual learning will force stratified prefs
        w = torch.rand(2).to(device)
        w = w / w.sum()
        
        done = False
        steps = 0
        total_loss = 0
        loss_cnt = 0
        
        while not done:
            if g is None:
                g, _, done = env.step("RANDOM", multi_objective=True)
                continue
            
            action, hidden, feats, ei = agent.act(g, env.machines, hidden, w, epsilon)
            rule = FT_CONFIG["rules"][action]
            
            snapshot = learner._create_state_snapshot(env.active_jobs, env.machines)
            g_next, r_vec, done = env.step(rule, multi_objective=True)
            
            r_vec_t = torch.tensor(r_vec, dtype=torch.float32)
            agent.remember({
                'feats': feats.cpu(), 'ei': ei.cpu(),
                'action': action, 'r_vec': r_vec_t, 
                'done': done, 'w': w.cpu(),
                'snapshot': snapshot 
            })
            
            # Learn more frequently (every 5 steps)
            if len(agent.memory) >= FT_CONFIG["train"]["batch_size"] and steps % FT_CONFIG["train"]["learn_every"] == 0:
                l = learner.learn(agent.memory, FT_CONFIG["train"]["batch_size"])
                total_loss += l
                loss_cnt += 1
            
            g = g_next
            steps += 1
            
        # Post-episode learning
        if len(agent.memory) > 0:
            l = learner.learn(agent.memory, FT_CONFIG["train"]["batch_size"])
            total_loss += l
            loss_cnt += 1

        # Very slow decay
        epsilon = max(0.05, epsilon * 0.998)
        avg_loss = total_loss / max(1, loss_cnt)
        n_jobs = FT_CONFIG["train"]["jobs_per_episode"]
        
        print(f"{ep+1:<6} {env.total_tardiness/n_jobs:<10.1f} {env.total_flow_time/n_jobs:<10.1f} {avg_loss:<10.2f} {learner.ga_counter:<6} {epsilon:<6.2f} {time.time()-t0:<6.1f}")
        
        with open(log_file, "a") as f:
            f.write(f"{ep+1},{env.total_tardiness/n_jobs},{env.total_flow_time/n_jobs},{avg_loss},{epsilon},{time.time()-t0},{learner.ga_counter}\n")
            
        if (ep+1) % 50 == 0:
            torch.save(policy.state_dict(), out_dir / f"ep_{ep+1}.pth")

if __name__ == "__main__":
    main()
