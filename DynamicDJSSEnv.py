# envs/DynamicDJSSEnv.py

import copy
import numpy as np
from sortedcontainers import SortedDict
from networks.CreatDisjunctiveGraph import creatDisjunctiveGraph
from herisDispRules import eval as rule_eval
from core.cOperation import Operation
from utils.scheduling_metrics import calculate_ttotal_single

class DynamicDJSSEnv:
    def __init__(self, job_generator, machines, device="cpu"):
        """
        :param job_generator: DynamicJobGenerator 实例
        :param machines: 机器列表模板
        :param device: 设备
        """
        self.job_gen = job_generator
        self.template_machines = machines 
        self.device = device
        
        # 核心状态变量
        self.active_jobs = []       
        self.machines = []          
        self.machine_events = SortedDict({0: 1}) 
        self.current_time = 0
        self.pending_arrivals = []  
        
        # 统计指标
        self.total_tardiness = 0
        self.total_flow_time = 0 # 新增：记录总流转时间
        self.finished_flow_time_sum = 0.0
        self.completed_jobs = []
        
    def reset(self, num_dynamic_jobs=50):
        self.current_time = 0
        self.machines = copy.deepcopy(self.template_machines)
        for m in self.machines:
            m.currentTime = 0
            m.assignedOpera = []
            
        self.machine_events = SortedDict({0: 1})
        self.job_gen.reset()
        
        self.pending_arrivals = self.job_gen.generate_arrival_sequence(num_dynamic_jobs)
        self.active_jobs = []
        
        self.total_tardiness = 0
        self.total_flow_time = 0
        self.finished_flow_time_sum = 0.0
        self.completed_jobs = []
        
        self._check_arrivals()
        
        return self._get_graph()

    def _calculate_current_total_flow_time(self):
        """新增：计算当前所有在制和已完成工件的流转时间总和"""
        total_flow = self.finished_flow_time_sum
        for j in self.active_jobs:
            flow = max(0, self.current_time - j.arrivalTime)
            total_flow += flow
        return total_flow

    def step(self, rule_name, multi_objective=False): # ✅ 新增参数
        # --- 1. 执行规则 ---
        scheduled_ops, _, self.machine_events = rule_eval(
            rule_name, self.active_jobs, self.machines, self.machine_events
        )
        
        # --- 2. 记录旧指标 ---
        prev_tardiness = self.total_tardiness
        prev_flow_time = self.total_flow_time # 使用成员变量

        # --- 3. 动态时间推进 ---
        done = False
        if not self.pending_arrivals and all(self._is_job_complete(j) for j in self.active_jobs):
            done = True
        else:
            next_machine_time = self.machine_events.peekitem(0)[0] if self.machine_events else float('inf')
            next_arrival_time = self.pending_arrivals[0].arrivalTime if self.pending_arrivals else float('inf')
            next_event_time = min(next_machine_time, next_arrival_time)
            
            if next_event_time != float('inf') and next_event_time > self.current_time:
                self.current_time = next_event_time
                
            self._check_arrivals()
            self._clean_completed_jobs()

        # --- 4. 计算新指标与 Reward ---
        curr_tardiness = self._calculate_current_total_tardiness()
        curr_flow_time = self._calculate_current_total_flow_time()

        delta_tardiness = curr_tardiness - prev_tardiness
        delta_flow_time = curr_flow_time - prev_flow_time
        
        self.total_tardiness = curr_tardiness
        self.total_flow_time = curr_flow_time # 更新成员变量

        if multi_objective:
            reward = [
                - (delta_tardiness / 1000.0),
                - (delta_flow_time / 2000.0)
            ]
            if len(scheduled_ops) > 0:
                reward[0] += 0.1
                reward[1] += 0.1
        else:
            reward = - (delta_tardiness / 1000.0)
            if len(scheduled_ops) > 0:
                reward += 0.1
        
        return self._get_graph(), reward, done

    def _check_arrivals(self):
        while self.pending_arrivals and self.pending_arrivals[0].arrivalTime <= self.current_time:
            job = self.pending_arrivals.pop(0)
            self.active_jobs.append(job)

    def _clean_completed_jobs(self):
        remaining_jobs = []
        for j in self.active_jobs:
            if self._is_job_complete(j):
                finish_time = max(op.endTime for op in j.operations)
                flow_time = finish_time - j.arrivalTime
                self.finished_flow_time_sum += flow_time
                self.completed_jobs.append({
                    "id": j.idItinerary,
                    "flow_time": flow_time,
                    "tardiness": max(0, finish_time - j.due_date)
                })
            else:
                remaining_jobs.append(j)
        self.active_jobs = remaining_jobs

    def _is_job_complete(self, job):
        if not job.operations: return True
        return all(op.completed for op in job.operations)

    def _get_graph(self):
        if not self.active_jobs: return None
        return creatDisjunctiveGraph(self.active_jobs, self.machines)

    def _calculate_current_total_tardiness(self):
        all_scheduled_ops = []
        for m in self.machines:
            if hasattr(m, 'assignedOpera'):
                all_scheduled_ops.extend(m.assignedOpera)
        return calculate_ttotal_single(all_scheduled_ops)
