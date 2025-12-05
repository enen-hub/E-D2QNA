# -*- coding: utf-8 -*-
import os
import random
from typing import List, Tuple, Dict
from sortedcontainers import SortedDict
import numpy as np  # for finite checks in sanity assertions

from networks.CreatDisjunctiveGraph import sameJob_op
from core.clItinerary import Itinerary as Job
from core.cOperation import Operation
from core.clMachine import Machine

class Individual:
    """
    遗传算法个体类
    用于NSGA-II的非支配排序和拥挤度计算
    """
    _uid_counter = 0
    
    def __init__(self, chromosome):
        self.chromosome = chromosome  # 染色体（可以是None）
        self.objectives = None        # 目标向量 [obj1, obj2, obj3]（负值空间）
        self.rank = 0                 # Pareto等级
        self.crowding_distance = 0.0  # 拥挤度
        self.uid = Individual._uid_counter  # 唯一ID
        Individual._uid_counter += 1
    
    def __repr__(self):
        return f"Individual(uid={self.uid}, rank={self.rank}, obj={self.objectives})"
        

def build_job_lookup(job_object_list: List[Job]) -> Dict[int, Job]:
    """按工件ID建立字典，便于把工件级 due_date 写回工序。"""
    return {int(getattr(job, "idItinerary")): job for job in job_object_list}

def _attach_job_due(op: Operation, job_lookup: Dict[int, Job]):
    try:
        parent = job_lookup[int(getattr(op, "idItinerary"))]
        op.parent_job = parent
        op.due_date = getattr(parent, "due_date", None)
    except Exception:
        pass

def find_previous_operation(task: Operation, sameTaskList_by_dict: dict):
    """在同一工件下，返回该工序的直接前序工序（若存在）。"""
    job_ops_list = sameTaskList_by_dict.get(task.itinerary)
    if not job_ops_list:
        return None
    if task.idOperation > 1:
        if len(job_ops_list) >= task.idOperation:
            return job_ops_list[task.idOperation - 2]
    return None

def _ensure_assigned_list(machine: Machine):
    if not hasattr(machine, "assignedOpera") or machine.assignedOpera is None:
        machine.assignedOpera = []

def _schedule_and_stamp_due(
    best_op: Operation,
    machine: Machine,
    current_time: float,
    time_dict: SortedDict,
    job_lookup: Dict[int, Job]
):
    # 写回 due
    try:
        job = job_lookup[int(getattr(best_op, "idItinerary"))]
        best_op.due_date = getattr(job, "due_date", None)
    except Exception:
        pass

    start_time = max(getattr(machine, "currentTime", 0.0), current_time)
    best_op.startTime = start_time
    best_op.completed = True
    best_op.assignedMachine = machine.name
    best_op.machine_power_running = getattr(machine, 'power_running', 3.0)
    best_op.machine_power_idle = getattr(machine, 'power_idle', 0.5)
    best_op.duration = best_op.machine[machine.name]
    best_op.endTime = best_op.startTime + best_op.duration

    machine.currentTime = best_op.endTime
    _ensure_assigned_list(machine)
    machine.assignedOpera.append(best_op)

    # 记录事件时刻
    if machine.currentTime not in time_dict:
        time_dict[machine.currentTime] = 1
    else:
        time_dict[machine.currentTime] += 1

def _collect_ready_candidates(
    all_operations_flat: List[Operation],
    machine: Machine,
    current_time: float,
    sameTaskList_dict: dict
) -> List[Operation]:
    candidates = []
    for task in all_operations_flat:
        if task.completed:
            continue
        if machine.name not in task.machine:
            continue
        # 就绪判定：首工序或前序已完且结束时间不晚于当前时刻
        is_ready = False
        if getattr(task, 'idOperation', 1) <= 1:
            is_ready = True
        else:
            pre_op = find_previous_operation(task, sameTaskList_dict)
            if pre_op is None:
                is_ready = True
            elif pre_op.completed and pre_op.endTime <= current_time + 1e-6:
                is_ready = True
        if is_ready:
            candidates.append(task)
    return candidates

def computingCompletedRatio(job_object_list: List[Job]) -> List[Job]:
    operasOfJob = sameJob_op(job_object_list)
    for job_name, tasks in operasOfJob.items():
        total = len(tasks)
        if total == 0:
            continue
        completed_num = sum(1 for opera in tasks if getattr(opera, "completed", False))
        ratio = completed_num / total
        for opera in tasks:
            opera.completedRatio = ratio
    return job_object_list


# =========================
# 启发式调度规则
# =========================

# === SPT: 最短加工时间优先 ===
def algorithmSPT(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    # ✅ 用“最新”的决策时刻
    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = min(candidates, key=lambda op: op.machine[machine.name])
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    # 事件推进：当前时刻处理完毕后移除
    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

   
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === LPT: 最长加工时间优先===
def algorithmLPT(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = max(candidates, key=lambda op: op.machine[machine.name])
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === SR: 最短剩余总时间优先（工件维度）===
def algorithmSR(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    def remaining_time_of_job(op: Operation) -> float:
        rem = 0.0
        job_ops = sameTaskList_dict.get(op.itinerary, [])
        for jop in job_ops:
            if not jop.completed:
                rem += min(jop.machine.values())
        return rem

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = min(candidates, key=remaining_time_of_job)
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === LR: 最长剩余总时间优先（工件维度）===
def algorithmLR(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    def remaining_time_of_job_max(op: Operation) -> float:
        rem = 0.0
        job_ops = sameTaskList_dict.get(op.itinerary, [])
        for jop in job_ops:
            if not jop.completed:
                rem += max(jop.machine.values())
        return rem

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = max(candidates, key=remaining_time_of_job_max)
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === FOPNR: 剩余工序数最少优先 ===
def algorithmFOPNR(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    def remaining_ops_of_job(op: Operation) -> int:
        job_ops = sameTaskList_dict.get(op.itinerary, [])
        return sum(1 for jop in job_ops if not jop.completed)

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = min(candidates, key=remaining_ops_of_job)
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === MORPNR: 剩余工序数最多优先 ===
def algorithmMORPNR(job_object_list: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(job_object_list)
    all_operations_flat = [op for job in job_object_list for op in getattr(job, 'operations', [])]

    currentTime = next(reversed(time)) if len(time) > 0 else 0.0
    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]

    job_lookup = build_job_lookup(job_object_list)

    def remaining_ops_of_job(op: Operation) -> int:
        job_ops = sameTaskList_dict.get(op.itinerary, [])
        return sum(1 for jop in job_ops if not jop.completed)

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = max(candidates, key=remaining_ops_of_job)
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    updated_job_list = computingCompletedRatio(job_object_list)
    return jobsListToExport, updated_job_list, SortedDict(time)


# === 随机分派 ===
def randomSolution(aJobsList: List[Job], machinesList: List[Machine], time: SortedDict) -> Tuple:
    import random as _r

    jobsListToExport: List[Operation] = []
    sameTaskList_dict = sameJob_op(aJobsList)
    currentTime = next(reversed(time)) if len(time) > 0 else 0.0

    idle_machines = [m for m in machinesList if m.currentTime <= currentTime + 1e-6]
    all_operations_flat = [op for job in aJobsList for op in getattr(job, 'operations', [])]

    job_lookup = build_job_lookup(aJobsList)

    for machine in idle_machines:
        candidates = _collect_ready_candidates(all_operations_flat, machine, currentTime, sameTaskList_dict)
        if not candidates:
            continue

        best_op = _r.choice(candidates)
        if id(best_op) in {id(op) for op in jobsListToExport}:
            continue

        _attach_job_due(best_op, job_lookup)
        _schedule_and_stamp_due(best_op, machine, currentTime, time, job_lookup)
        jobsListToExport.append(best_op)

    if currentTime in time:
        del time[currentTime]
    if not jobsListToExport and not time:
        non_idle = [m for m in machinesList if m.currentTime > currentTime]
        if non_idle:
            nxt = min(m.currentTime for m in non_idle)
            if nxt not in time:
                time[nxt] = 1

    
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for m in machinesList:
            _ensure_assigned_list(m)
            assert np.isfinite(m.currentTime), "Machine currentTime must be finite"
            last_end = -float("inf")
            for op in m.assignedOpera:
                assert hasattr(op, "startTime") and hasattr(op, "endTime"), "Op missing times"
                assert hasattr(op, "assignedMachine"), "Op missing assignedMachine"
                assert op.startTime <= op.endTime + 1e-12, "Op start>end"
                assert op.assignedMachine == m.name, "Op assignedMachine mismatch"
                assert getattr(op, "completed", True), "Exported op should be completed"
                assert op.startTime >= last_end - 1e-9, "Machine sequence time not non-decreasing"
                last_end = op.endTime

    aJobsList = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, SortedDict(time)



dispatchingRules = {
    "SPT": algorithmSPT,
    "LPT": algorithmLPT,
    "SR": algorithmSR,
    "LR": algorithmLR,
    "FOPNR": algorithmFOPNR,
    "MORPNR": algorithmMORPNR,
    "RANDOM": randomSolution
}

# 评估函数，用于评估调度规则
def eval(rule_name, job_list, machine_list, time_dict):

    if rule_name in dispatchingRules:
        return dispatchingRules[rule_name](job_list, machine_list, time_dict)
    else:
        raise ValueError(f"未知的调度规则: {rule_name}")

