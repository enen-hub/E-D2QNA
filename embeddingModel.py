import torch
import copy
from networks.CreatDisjunctiveGraph import creatDisjunctiveGraph
import contextlib
from collections import defaultdict
from typing import Any, Iterable
import re
import os


def _to_torch_device(device):
    try:
        return device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _is_cuda_device(device):
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device).startswith('cuda')


# ===============================
# Node Feature Extraction
# ===============================
def get_features_of_node(disjunctive_graph, machine_list, device="cuda:0"):
    node_features = []
    machine_time_map = {getattr(m, "name", f"M{i}"): float(getattr(m, "currentTime", 0.0)) for i, m in enumerate(machine_list)}
    all_features = []

    for node_name in disjunctive_graph.nodes:
        node_data = disjunctive_graph.nodes[node_name]

        if node_name in ("S", "E"):
            node_type = [1, 0, 0] if node_name == "S" else [0, 1, 0]
            feat = node_type + [0, 0, 0, 0, 0, 0, 0, 0]  # 补齐到11维
            all_features.append(feat)
            continue

        op = node_data.get('keyword', None)
        if op is None or not hasattr(op, 'idOperation'):
            continue

        # Regular operation node
        node_type = [0, 0, 1]
        completed = 1.0 if bool(getattr(op, "completed", False)) else 0.0
        priority = float(getattr(op, "priority", 0.0))
        id_op = float(getattr(op, "idOperation", 0))
        id_job = float(getattr(op, "idItinerary", 0))

        duration = float(getattr(op, "duration", 0.0))
        assigned_machine = getattr(op, "assignedMachine", None)
        assigned_flag = 1.0 if assigned_machine is not None else 0.0
        denom = float(machine_time_map.get(assigned_machine, 0.0)) + 1e-6
        dur_over_mt = float(duration / denom) if assigned_machine is not None else float(duration)
        mach_cnt = float(len(getattr(op, "machine", {}) or {}))

        feat = node_type + [
            completed,
            priority,
            id_op,
            id_job,
            duration,
            dur_over_mt,
            mach_cnt,
            assigned_flag
        ]
        all_features.append(feat)

    if not all_features:
        raise ValueError("No valid operation nodes found in disjunctive graph!")

    dev = torch.device(device) if isinstance(device, str) else device
    feat_array = torch.tensor(all_features, dtype=torch.float32, device=dev)

    # 列归一化
    feat_min = torch.min(feat_array, dim=0, keepdim=False)[0]
    feat_max = torch.max(feat_array, dim=0, keepdim=False)[0]
    feat_range = torch.where(feat_max == feat_min, torch.tensor(1.0, device=dev), feat_max - feat_min)

    normalized_feats = (feat_array - feat_min) / feat_range
    
 
    normalized_feats = torch.nan_to_num(
        normalized_feats,
        nan=0.5,   # 中性值，避免把"未知"当作"最小"
        posinf=1.0,
        neginf=0.0
    )

   
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        assert torch.isfinite(normalized_feats).all(), "Node features contain non-finite values"
        assert normalized_feats.dim() == 2, "Feature shape mismatch"
        # 固定特征维度为11（与模型 raw_feat_dim 对齐）
        assert normalized_feats.size(1) == 11, f"Expected 11 feature columns, got {normalized_feats.size(1)}"

    return normalized_feats



# ===============================
# Adjacency Matrix
# ===============================
def get_adjacent_matrix(graph, device=None, weighted: bool = False, debug: bool = False):

    import torch

    nodes = list(graph.nodes)
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 设备更稳健转换
    dev = _to_torch_device(device) if device is not None else torch.device('cpu')
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=dev)

    conj_count = 0
    disj_count = 0
    edges_processed = 0

    for u, v, k, data in graph.edges(keys=True, data=True):
        try:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
        except KeyError:
            if debug:
                print(f"[WARN] Edge endpoint not found in nodes: {u}->{v}")
            continue

        edge_type = data.get('type', None)
        processing_time = float(data.get('processing_time', 0.0))

        if not weighted:
            
            adj[u_idx, v_idx] = 1.0
        else:
                       
            if edge_type == 'CONJUNCTIVE':
                # 用0.1而非eps，确保工序顺序约束在GCN中可见
                adj[u_idx, v_idx] = 0.1
                conj_count += 1
            elif edge_type == 'DISJUNCTIVE':
                # DISJUNCTIVE用实际加工时间
                adj[u_idx, v_idx] = max(processing_time, 0.1)
                disj_count += 1
            else:
                # 未知类型保守处理
                adj[u_idx, v_idx] = max(processing_time, 0.1)

        edges_processed += 1

    if debug:
        nonzero = (adj != 0).sum().item()
        print("\n[ADJ SUMMARY]")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges processed: {edges_processed}")
        if weighted:
            print(f"  Conjunctive edges (0.1): {conj_count}")
            print(f"  Disjunctive edges (pt): {disj_count}")
        print(f"  Non-zero entries: {int(nonzero)}")

    # 轻量自检
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        assert adj.shape[0] == adj.shape[1] == num_nodes, "Adjacency shape mismatch"
        assert torch.isfinite(adj).all(), "Adjacency contains non-finite values"

    return adj