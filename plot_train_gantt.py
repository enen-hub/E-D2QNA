import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
import sys
import os
from pathlib import Path
from sortedcontainers import SortedDict

# --- 引入项目路径 ---
try:
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from networks.PolicyNet import Policy
from core.DynamicJobGenerator import DynamicJobGenerator
from envs.DynamicDJSSEnv import DynamicDJSSEnv
from utils.data_loader import readData
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix

# --- 配置 ---
# 替换为你想要可视化的模型路径
MODEL_PATH = "result/E_D2QNA_FINAL_V2/ep_400.pth" 
JSON_PATH = "dataset/la16/la16_K1.3.json"
URGENT_PROB = 0.3

# 规则映射
RULE_MAP = ["SPT", "LPT", "FOPNR", "MORPNR", "SR", "LR", "RANDOM"]

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = {
        "model": {
            "raw_feat_dim": 11,
            "num_actions": 7,
            "embed_dim": 32,
            "gcn_hidden": 256,
            "num_heads": 8,
            "lstm_hidden": 128,
            "num_objectives": 2
        },
        "action_space": {"num_rules": 7},
        "objectives_config": {"active_objectives": ["tardiness", "flow_time"]}
    }
    policy = Policy(config, device=str(device))
    state = checkpoint.get('model_state_dict', checkpoint)
    current = policy.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    policy.load_state_dict(filtered, strict=False)
    policy.eval()
    return policy

def run_simulation(policy, json_path, device):
    _, jobs, machines = readData(json_path, validate=False)
    job_gen = DynamicJobGenerator(jobs, config={"seed": 42, "urgent_prob": URGENT_PROB})
    env = DynamicDJSSEnv(job_gen, machines, device=device)
    
    w = torch.tensor([1.0, 0.0], device=device)
    
    g = env.reset(num_dynamic_jobs=50)
    hidden = policy.init_hidden_state(1)
    done = False
    
    while not done:
        if g is None:
            g, _, done = env.step("RANDOM", multi_objective=True)
            continue
            
        with torch.no_grad():
            feats = get_features_of_node(g, env.machines, device=device)
            adj = get_adjacent_matrix(g, device=device)
            ei = adj.to_sparse().indices()
            
            w_input = w.view(1, -1)
            try:
                q, hidden = policy(feats, ei, hidden, batch=None, w=w_input)
            except:
                feats_cat = torch.cat([feats, w_input.repeat(feats.size(0), 1)], dim=1)
                q, hidden = policy(feats_cat, ei, hidden)

            score = (q * w.view(1,1,-1)).sum(dim=2)
            action = score.argmax().item()
            
        rule = RULE_MAP[action]
        g, _, done = env.step(rule, multi_objective=True)
        
    return env.machines

def draw_gantt(machines):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 生成颜色池 (每个 Job 一个颜色)
    colors = plt.cm.tab20.colors
    
    y_ticks = []
    y_labels = []
    
    for i, m in enumerate(machines):
        y = i * 10
        y_ticks.append(y + 5)
        y_labels.append(m.name)
        
        # 遍历机器上的所有工序
        for op in m.assignedOpera:
            start = op.startTime
            duration = op.duration
            job_id = int(getattr(op.parent_job, 'idItinerary'))
            color = colors[job_id % len(colors)]
            
            # 画矩形
            rect = patches.Rectangle((start, y), duration, 9, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
            # 标注 Job ID
            ax.text(start + duration/2, y + 4.5, f"J{job_id}", ha='center', va='center', fontsize=8, color='white')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_title('Scheduling Gantt Chart (E-D2QNA)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # 自动调整 X 轴范围
    max_time = max([m.currentTime for m in machines])
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(0, len(machines) * 10)
    
    plt.tight_layout()
    plt.savefig('schedule_gantt.png', dpi=300)
    print("✅ Gantt chart saved to schedule_gantt.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Simulating schedule...")
    policy = load_model(MODEL_PATH, device)
    machines = run_simulation(policy, JSON_PATH, device)
    print("🎨 Drawing Gantt chart...")
    draw_gantt(machines)
