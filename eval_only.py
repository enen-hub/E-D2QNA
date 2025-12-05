# -*- coding: utf-8 -*-
"""
Eval-only script for generating Pareto Front analysis.
Fixed for E-D2QNA Final Config (feat_dim=13, gcn=256, etc.)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from sortedcontainers import SortedDict
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Categorical

# Project root setup
try:
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# Import your modules
# NOTE: Ensure config.py exists or PARAMS_MULTI_OBJECTIVE is available
try:
    from config import PARAMS_MULTI_OBJECTIVE
except ImportError:
    # Fallback config if config.py missing
    PARAMS_MULTI_OBJECTIVE = {
        "model": {
            "raw_feat_dim": 13,  # ✅ Corrected for your final model
            "num_actions": 7,
            "embed_dim": 32,
            "gcn_hidden": 256,   # ✅ Corrected
            "num_heads": 8,
            "lstm_hidden": 128,  # ✅ Corrected
            "num_objectives": 2
        },
        "action_space": {"num_rules": 7},
        "objectives_config": {"active_objectives": ["tardiness", "flow_time"]}
    }

from networks.PolicyNet import Policy
from utils.data_loader import readData
from networks.CreatDisjunctiveGraph import creatDisjunctiveGraph
from networks.embeddingModel import get_features_of_node, get_adjacent_matrix
from utils.scheduling_metrics import calculate_objectives
from herisDispRules import dispatchingRules

# Optional HV indicator
try:
    from pymoo.indicators.hv import HV
except Exception:
    HV = None

# Stratified preference grid (2-objective)
PREF_GRID = [
    [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0]
]

def _prep_state(jobs, machines, device: str):
    g = creatDisjunctiveGraph(jobs, machines)
    adj = get_adjacent_matrix(g, device=device) # Pass device to adj gen
    edge_index, _ = dense_to_sparse(adj)
    feats = get_features_of_node(g, machines, device=device).to(device)
    edge_index = edge_index.to(device)
    return {"features": feats, "edge_index": edge_index}

def _align_features(feats: torch.Tensor, model: Policy, pref: torch.Tensor) -> torch.Tensor:
    # Dynamically align feature dimension with model expectation
    need = int(getattr(model.feat_embed, 'in_features', 13)) # Default 13
    cur = int(feats.shape[1])
    
    if cur < need:
        # Pad with preference vector
        w = pref.view(1, -1).repeat(feats.shape[0], 1)
        pad = w
        # Handle case where pref dim doesn't perfectly fill gap
        if (cur + w.shape[1]) > need:
            pad = w[:, : (need - cur)]
        elif (cur + w.shape[1]) < need:
            # If still not enough, pad with zeros (rare case)
            zeros = torch.zeros((feats.shape[0], need - cur - w.shape[1]), device=feats.device)
            pad = torch.cat([w, zeros], dim=1)
            
        feats = torch.cat([feats, pad], dim=1)
    elif cur > need:
        feats = feats[:, :need]
    return feats

def _choose_action(policy: Policy, state_feat: dict, hidden, pref_vec: torch.Tensor, temp: float,
                   strategy: str = "softmax") -> tuple:
    with torch.no_grad():
        q_vectors, new_hidden = policy(state_feat["features"], state_feat["edge_index"], hidden)
        # Scalarize Q-values using preference w
        scores = torch.sum(q_vectors * pref_vec.view(1, 1, -1), dim=2).float().squeeze(0)
        
        logits = scores
        if temp is not None and temp > 0:
            logits = logits / float(max(temp, 1e-6))
            
        if strategy == "softmax":
            probs = torch.softmax(logits, dim=-1)
            act = int(Categorical(probs).sample().item())
        else:
            act = int(torch.argmax(logits, dim=0).item())
            
    return act, new_hidden

def final_rollout_once(policy: Policy, jobs, machines, preference: torch.Tensor, temp: float, device: str) -> list:
    pro_time = SortedDict({0: 1})
    hidden = policy.init_hidden_state(batch_size=1)
    completed = 0
    total_ops = sum(len(j.operations) for j in jobs)
    
    # Filter rules (exclude RANDOM if needed, but keeping consistent with training is safer)
    # Using the standard rule set
    rule_keys = ['SPT', 'LPT', 'SR', 'LR', 'FOPNR', 'MORPNR', 'RANDOM']
    
    while completed < total_ops:
        state = _prep_state(jobs, machines, device)
        state["features"] = _align_features(state["features"], policy, preference)
        
        act, hidden = _choose_action(policy, state, hidden, preference, temp)
        
        rule_name = rule_keys[act % len(rule_keys)]
        jobs_exported, jobs, pro_time = dispatchingRules[rule_name](jobs, machines, pro_time)
        completed += len(jobs_exported)

    schedule = [op for m in machines for op in getattr(m, "assignedOpera", [])]
    if schedule:
        # Use calculate_objectives from utils
        # Note: Ensure your PARAMS_MULTI_OBJECTIVE has 'objectives_config'
        obj = calculate_objectives(schedule=schedule, machines=machines, config=PARAMS_MULTI_OBJECTIVE, mode="final", device=device)
        # Return [Cmax, Ttotal]
        return [float(obj.get("cmax", np.inf)), float(obj.get("ttotal", np.inf))]
    return [float('inf'), float('inf')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--instance", type=str, default="dataset/la16/la16_K1.3.json")
    parser.add_argument("--output_dir", type=str, default="result/final_analysis")
    parser.add_argument("--final_rollouts", type=int, default=200)
    parser.add_argument("--stratified_sampling", action="store_true", help="Use grid preferences")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 1. Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔥 Starting Pareto Analysis on {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model (Robustly)
    print(f"🔄 Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Force use the correct config for your final model
    config = PARAMS_MULTI_OBJECTIVE
    # Ensure critical dims match your training script
    config["model"]["raw_feat_dim"] = 13
    config["model"]["gcn_hidden"] = 256
    config["model"]["lstm_hidden"] = 128
    
    policy = Policy(config, device=str(device))
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        policy.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully (Strict).")
    except Exception as e:
        print(f"⚠️ Strict load failed: {e}")
        print("🔄 Attempting non-strict load...")
        policy.load_state_dict(state_dict, strict=False)
        
    policy.eval()
    policy.to(device)

    # 3. Load Data
    instance_path = args.instance
    if not os.path.exists(instance_path):
        # Try local fallback
        fallback = Path(PROJECT_ROOT) / instance_path
        if fallback.exists():
            instance_path = str(fallback)
        else:
            print(f"❌ Instance not found: {instance_path}")
            return
            
    print(f"📂 Loading instance: {instance_path}")
    _, jobs_tmpl, machines_tmpl = readData(instance_path, is_batch=False, validate=False)

    # 4. Evaluation Loop
    points = []
    
    # Prepare preferences
    if args.stratified_sampling:
        pref_list = [torch.tensor(w, dtype=torch.float32, device=device) for w in PREF_GRID]
        print(f"📊 Using Stratified Sampling: {len(pref_list)} grid points")
    else:
        pref_list = [] # Will generate random on the fly
        print("🎲 Using Random Sampling")

    total = args.final_rollouts
    for i in range(total):
        # Reset Env
        jobs = deepcopy(jobs_tmpl)
        machines = deepcopy(machines_tmpl)
        
        # Select preference
        if args.stratified_sampling:
            pref = pref_list[i % len(pref_list)]
        else:
            w = np.random.dirichlet([1.0, 1.0])
            pref = torch.tensor(w, dtype=torch.float32, device=device)
            
        # Run
        # Use low temperature for exploitation
        pt = final_rollout_once(policy, jobs, machines, pref, temp=0.1, device=str(device))
        points.append(pt)
        
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{total}", end="\r")
            
    print("\n✅ Rollouts finished.")

    # 5. Process Results (Pareto Front)
    from utils.pareto import non_dominated_filter_min, dedup_tolerant
    
    points_arr = np.array(points)
    # Remove infs
    valid_mask = np.isfinite(points_arr).all(axis=1)
    points_arr = points_arr[valid_mask]
    
    if len(points_arr) == 0:
        print("❌ No valid solutions found!")
        return

    # Filter Non-Dominated
    unique_pts = dedup_tolerant(points_arr, decimals=(0, 0))
    pf = non_dominated_filter_min(unique_pts)
    
    # Calculate HV (Simple version)
    hv = 0.0
    if HV:
        # Normalize for HV calc (assuming range [0, 10000])
        try:
            ref_point = np.array([1.1, 1.1])
            # Min-Max Normalization approx
            mins = points_arr.min(axis=0)
            maxs = points_arr.max(axis=0)
            if (maxs > mins).all():
                norm_pf = (pf - mins) / (maxs - mins)
                ind = HV(ref_point=ref_point)
                hv = ind.do(norm_pf)
        except:
            pass

    print(f"\n🏆 Final Results:")
    print(f"   PF Size: {len(pf)}")
    print(f"   Hypervolume: {hv:.4f}")
    
    # 6. Save & Plot
    np.save(out_dir / "pf_points.npy", pf)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    
    # Plot all candidates (grey)
    plt.scatter(points_arr[:, 0], points_arr[:, 1], c='gray', alpha=0.3, label='Candidates')
    # Plot PF (red)
    pf_sorted = pf[np.argsort(pf[:, 0])]
    plt.plot(pf_sorted[:, 0], pf_sorted[:, 1], c='red', linewidth=2, label='Pareto Front')
    plt.scatter(pf_sorted[:, 0], pf_sorted[:, 1], c='red', s=50)
    
    plt.xlabel('Makespan (Cmax)')
    plt.ylabel('Total Tardiness')
    plt.title(f'Pareto Front Analysis (Model: {Path(args.model_path).stem})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = out_dir / "pareto_front.png"
    plt.savefig(plot_path, dpi=300)
    print(f"🖼️ Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()