from collections import defaultdict
import os
import numpy as np
import torch


# ---------- 工具函数 ----------

def _safe_to_float(x, default=0.0) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


def _get_job_level_due(op) -> float:
    
    dd = getattr(op, "due_date", None)
    if dd is not None:
        return _safe_to_float(dd, float('inf'))
    parent = getattr(op, "parent_job", None)
    if parent is not None and getattr(parent, "due_date", None) is not None:
        return _safe_to_float(parent.due_date, float('inf'))
    return float('inf')


# ========================================
# 目标1: Cmax (Makespan)
# ========================================

def calculate_cmax_single(final_schedule: list, device: str = "cpu") -> float:
    if not final_schedule:
        return 0.0
    end_times = [
        op.endTime for op in final_schedule
        if hasattr(op, 'endTime') and op.endTime is not None
    ]
    if not end_times:
        return 0.0
    return float(torch.tensor(end_times, dtype=torch.float32, device=device).max().item())


# ========================================
# 目标2: Ttotal (Total Tardiness)
# ========================================

def calculate_ttotal_single(schedule: list, device: str = "cpu") -> float:
    if not schedule:
        return 0.0

    job_metrics = defaultdict(lambda: {'latest_end_time': 0.0, 'due_date': float('inf')})

    for op in schedule:
        jid = getattr(op, "idItinerary", None)
        if jid is None:
            continue
        end_time = _safe_to_float(getattr(op, "endTime", 0.0), 0.0)
        if end_time > job_metrics[jid]['latest_end_time']:
            job_metrics[jid]['latest_end_time'] = end_time
        job_metrics[jid]['due_date'] = min(job_metrics[jid]['due_date'], _get_job_level_due(op))

    total = 0.0
    for _, m in job_metrics.items():
        dd = m['due_date']
        if np.isfinite(dd):
            total += max(0.0, m['latest_end_time'] - dd)
    return float(total)


def calculate_ttotal_lookahead(final_schedule: list, device: str = "cpu", 
                                sim_time: float = 0.0) -> float:
    if not final_schedule:
        return 0.0

    job_metrics = defaultdict(lambda: {
        'earliest': float('inf'),
        'latest': 0.0,
        'due': float('inf'),
        'sched_ops': 0,
        'total_ops': 1
    })

    for op in final_schedule:
        jid = getattr(op, 'idItinerary', None)
        if jid is None:
            continue

        m = job_metrics[jid]
        
        if getattr(op, 'endTime', None) is not None:
            m['latest'] = max(m['latest'], _safe_to_float(op.endTime, 0.0))
        if getattr(op, 'startTime', None) is not None:
            m['earliest'] = min(m['earliest'], _safe_to_float(op.startTime, 0.0))

        m['sched_ops'] += 1
        m['due'] = min(m['due'], _get_job_level_due(op))

        parent = getattr(op, 'parent_job', None)
        if parent is not None and getattr(parent, 'operations', None):
            m['total_ops'] = max(1, len(parent.operations))

    total_delay = 0.0
    for m in job_metrics.values():
        completion = m['sched_ops'] / max(1, m['total_ops'])
        expected_time = (m['due'] if np.isfinite(m['due']) else 0.0) * completion
        earliest = 0.0 if m['earliest'] == float('inf') else m['earliest']
        actual_time = m['latest'] - earliest
        total_delay += max(0.0, actual_time - expected_time)

    return float(total_delay)


# TEC已移除


# ========================================
# 统一计算接口（用于训练和评估）
# ========================================

def calculate_objectives(schedule: list, machines: list, config: dict, 
                         mode: str = "final", sim_time: float = 0.0,
                         device: str = "cpu") -> dict:
    """
    ===== 统一的目标计算接口（双目标：Cmax + Ttotal） =====
    
    参数：
        schedule: 调度方案
        machines: 机器列表
        config: 配置字典
        mode: "final" | "lookahead"
            - "final": 使用真实指标（最终评估）
            - "lookahead": 使用代理指标（GA短窗评估）
        device: 计算设备
    
    返回：
        {
            "cmax": float (正值),
            "ttotal": float (正值)
        }
    """
    obj_config = config.get("objectives_config", {})
    
    # Cmax 始终用相同计算方式
    cmax = calculate_cmax_single(schedule, device=device)
    
    # Ttotal 根据模式选择
    if mode == "lookahead":
        ttotal = calculate_ttotal_lookahead(schedule, device=device, sim_time=sim_time)
    else:  # final
        ttotal_mode = obj_config.get("ttotal_mode", "true_tardiness")
        if ttotal_mode == "true_tardiness":
            ttotal = calculate_ttotal_single(schedule, device=device)
        else:
            ttotal = calculate_ttotal_lookahead(schedule, device=device, sim_time=sim_time)
    
    # 双目标返回
    return {
        "cmax": float(cmax),
        "ttotal": float(ttotal)
    }