# -*- coding: utf-8 -*-
import numpy as np
import copy
import random
from core.clItinerary import Itinerary
from core.cOperation import Operation

class DynamicJobGenerator:
    def __init__(self, job_templates, config=None):
        """
        :param job_templates: 从 readData 读取的静态工件列表（作为模板，用于克隆工序结构）
        :param config: 配置字典，支持 urgent_prob (紧急任务概率)
        """
        self.templates = job_templates
        # 如果没有传入 seed，默认使用 42
        self.config = config or {}
        seed = self.config.get("seed", 42)
        self.np_random = np.random.RandomState(seed)
        
        # --- ETDQN 核心参数 (基础流量) ---
        self.mu = 88    
        self.PM = 0.5   
        self.P = 0.95   
        
        # --- [NEW] 困难模式参数 ---
        # 紧急长任务的生成概率 (默认 0.0，即关闭。消融实验中设为 0.2 或 0.3)
        self.urgent_prob = self.config.get("urgent_prob", 0.0) 
        
        self.job_counter = 0

    def reset(self):
        """重置计数器"""
        self.job_counter = 0

    def _interArrival(self):
        """ETDQN 基础到达间隔"""
        lamada = self.mu * self.PM / self.P
        inter_arrival = self.np_random.poisson(lamada)
        while inter_arrival == 0:
            inter_arrival = self.np_random.poisson(lamada)
        return inter_arrival

    def generate_arrival_sequence(self, num_jobs, start_time=0):
        """
        生成带有 '陷阱工件' 的到达序列
        """
        arrivals = []
        current_time = start_time
        
        for _ in range(num_jobs):
            # 1. 计算到达时间 (基础泊松流)
            dt = self._interArrival()
            current_time += dt
            
            # 2. 随机选择一个模板
            tmpl = self.np_random.choice(self.templates)
            
            # 3. [关键修改] 判定是否生成 '陷阱工件' (Urgent Long Job)
            is_urgent_long = self.np_random.rand() < self.urgent_prob
            
            if is_urgent_long:
                # 陷阱特征：
                # A. 工时长 (Scale = 3.0 ~ 5.0 倍) -> SPT 不愿意做
                # B. 交期紧 (Factor = 1.1 ~ 1.2) -> 必须马上做，否则严重拖期
                proc_scale = self.np_random.uniform(3.0, 5.0)
                dd_factor = self.np_random.uniform(1.1, 1.2)
                job_tag = "URGENT"
            else:
                # 普通工件
                proc_scale = 1.0
                dd_factor = self.np_random.uniform(1.3, 1.6)
                job_tag = "REGULAR"

            self.job_counter += 1
            new_job_name = f"DynJob_{self.job_counter}_{job_tag}"
            
            # 4. 计算缩放后的总工时并设定交期
            total_proc_time = 0
            for op in tmpl.operations:
                # 兼容性获取机器时间
                raw_times = getattr(op, 'machine', None) or getattr(op, 'candidate_machines', {})
                if raw_times:
                    # 取最小值 * 缩放倍数
                    total_proc_time += min(raw_times.values()) * proc_scale
            
            due_date = current_time + total_proc_time * dd_factor
            
            # 创建 Job 对象
            new_job = Itinerary(new_job_name, int(due_date))
            # 使用 .id 属性 (兼容性)
            new_job.id = str(self.job_counter + 10000)
            new_job.arrivalTime = int(current_time)
            new_job.operations = []
            
            # 5. 复制工序并应用缩放
            for tmpl_op in tmpl.operations:
                raw_machines = getattr(tmpl_op, 'machine', None) or getattr(tmpl_op, 'candidate_machines', {})
                
                # [关键] 对所有候选机器的加工时间应用 scale
                scaled_machines = {}
                for m_name, t_val in raw_machines.items():
                    scaled_machines[m_name] = max(1, int(t_val * proc_scale))
                
                new_op = Operation(
                    parent_job=new_job,
                    operation_id=tmpl_op.idOperation,
                    candidate_machines=scaled_machines
                )
                
                # 强制赋值 .machine 属性
                new_op.machine = scaled_machines
                new_op.due_date = int(due_date)
                new_op.idItinerary = new_job.idItinerary
                
                if hasattr(tmpl_op, 'priority'):
                    new_op.priority = tmpl_op.priority
                    
                new_job.operations.append(new_op)
                
            arrivals.append(new_job)
            
        return arrivals
