# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import networkx as nx
# 延迟导入 matplotlib 以避免训练时的非必要依赖成本
from collections import OrderedDict, defaultdict
from sortedcontainers import SortedDict
from core.clItinerary import Itinerary
from core.cOperation import Operation
from core.clMachine import Machine

# Edge type constants
CONJUNCTIVE_TYPE = 0
DISJUNCTIVE_TYPE = 1
FORWARD = 0
BIDIRECTION = 1


def _ensure_machine_assigned_list(machinesList):
    for m in machinesList:
        if not hasattr(m, "assignedOpera") or m.assignedOpera is None:
            m.assignedOpera = []


def creatDisjunctiveGraph(job_list: list[Itinerary], machinesList: list[Machine]) -> nx.MultiDiGraph:
    _ensure_machine_assigned_list(machinesList)  
    g = nx.MultiDiGraph()

    all_operations = [op for job in job_list for op in job.operations]
    waiting_operations = [op for op in all_operations if not op.completed]

    # 建点
    for op in all_operations:
        node_name = f"O{op.idItinerary}{op.idOperation}"
        g.add_node(node_for_adding=node_name, keyword=op)

    # 未完工操作的“同机候选集”
    disjunctive_tasks = disjunctive_ops(waiting_operations, machinesList)

    # 同工件工序（有序）
    sameJobOpts = sameJob_op(job_list)

    # 1) 工件内前后约束：CONJUNCTIVE（op_k -> op_{k+1}）
    for job_name, operations in sameJobOpts.items():
        for i in range(len(operations) - 1):
            op1 = operations[i]
            op2 = operations[i + 1]
            node1_name = f"O{op1.idItinerary}{op1.idOperation}"
            node2_name = f"O{op2.idItinerary}{op2.idOperation}"

            if node1_name not in g:
                g.add_node(node_for_adding=node1_name, keyword=op1)
            if node2_name not in g:
                g.add_node(node_for_adding=node2_name, keyword=op2)

            g.add_edge(
                node1_name,
                node2_name,
                type='CONJUNCTIVE',
                processing_time=0.0,
                direction=FORWARD,
                internal_type=CONJUNCTIVE_TYPE
            )

    # 2) 机器顺序约束：已分派顺序 + 已分派尾 -> 等待集（DISJUNCTIVE）
    for machine in machinesList:
        assiOperas = sorted(machine.assignedOpera, key=lambda op: getattr(op, "startTime", 0.0))
        # 已分派顺序链
        if len(assiOperas) > 1:
            for i in range(len(assiOperas) - 1):
                op1 = assiOperas[i]
                op2 = assiOperas[i + 1]
                node1_name = f"O{op1.idItinerary}{op1.idOperation}"
                node2_name = f"O{op2.idItinerary}{op2.idOperation}"
                proc_time = float(getattr(op1, "duration", 0.0)) if getattr(op1, "duration", 0.0) > 0 else 0.0

                if node1_name not in g:
                    g.add_node(node_for_adding=node1_name, keyword=op1)
                if node2_name not in g:
                    g.add_node(node_for_adding=node2_name, keyword=op2)

                g.add_edge(
                    node1_name,
                    node2_name,
                    type='DISJUNCTIVE',
                    processing_time=proc_time,
                    direction=FORWARD,
                    machine=machine.name,
                    internal_type=DISJUNCTIVE_TYPE
                )

        # 已分派尾 -> 机器等待集
        if assiOperas:
            last_assigned_op = assiOperas[-1]
            last_node_name = f"O{last_assigned_op.idItinerary}{last_assigned_op.idOperation}"
            last_proc_time = float(getattr(last_assigned_op, "duration", 0.0)) if getattr(last_assigned_op, "duration", 0.0) > 0 else 0.0

            for waiting_op in disjunctive_tasks.get(machine.name, []):
                waiting_node_name = f"O{waiting_op.idItinerary}{waiting_op.idOperation}"

                if last_node_name not in g:
                    g.add_node(node_for_adding=last_node_name, keyword=last_assigned_op)
                if waiting_node_name not in g:
                    g.add_node(node_for_adding=waiting_node_name, keyword=waiting_op)

                g.add_edge(
                    last_node_name,
                    waiting_node_name,
                    type='DISJUNCTIVE',
                    processing_time=last_proc_time,
                    direction=FORWARD,
                    machine=machine.name,
                    internal_type=DISJUNCTIVE_TYPE
                )

        # 机器等待集的两两冲突（双向）
        waiting_ops_on_machine = disjunctive_tasks.get(machine.name, [])
        for i in range(len(waiting_ops_on_machine)):
            for j in range(i + 1, len(waiting_ops_on_machine)):
                op1 = waiting_ops_on_machine[i]
                op2 = waiting_ops_on_machine[j]
                node1_name = f"O{op1.idItinerary}{op1.idOperation}"
                node2_name = f"O{op2.idItinerary}{op2.idOperation}"

                proc_time_12 = float(op1.machine.get(machine.name, 0))
                proc_time_21 = float(op2.machine.get(machine.name, 0))

                if node1_name not in g:
                    g.add_node(node_for_adding=node1_name, keyword=op1)
                if node2_name not in g:
                    g.add_node(node_for_adding=node2_name, keyword=op2)

                # 双向边
                g.add_edge(
                    node1_name,
                    node2_name,
                    type='DISJUNCTIVE',
                    processing_time=proc_time_12,
                    direction=BIDIRECTION,
                    machine=machine.name,
                    internal_type=DISJUNCTIVE_TYPE
                )
                g.add_edge(
                    node2_name,
                    node1_name,
                    type='DISJUNCTIVE',
                    processing_time=proc_time_21,
                    direction=BIDIRECTION,
                    machine=machine.name,
                    internal_type=DISJUNCTIVE_TYPE
                )

    _sanity_check_graph(g)  # [SANITY]
    # 轻量断言，确保图非空且边属性完整（调试模式）
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        assert g.number_of_nodes() > 0, "Disjunctive graph has no nodes"
        # 检查任意边是否包含期望的属性键
        for _, _, _, data in g.edges(keys=True, data=True):
            assert 'type' in data, "Edge missing 'type' attribute"
            assert 'processing_time' in data, "Edge missing 'processing_time' attribute"
    return g


def disjunctive_ops(operation_list: list[Operation], machineList: list[Machine]) -> dict[str, list[Operation]]:
    disjunctive_ops_map = defaultdict(list)
    for task in operation_list:
        mdict = getattr(task, "machine", {}) or {}
        for machine_name in mdict:
            disjunctive_ops_map[machine_name].append(task)
    return disjunctive_ops_map


def sameJob_op(job_object_list: list[Itinerary]) -> dict:
    sameJobOpts = {}
    for job_obj in job_object_list:
        if job_obj.operations:
            sorted_ops = sorted(job_obj.operations, key=lambda op: op.idOperation)
            # 以 name 为键（保持你原来的用法）
            sameJobOpts[job_obj.name] = sorted_ops
            # 额外：以 idItinerary 为键（防止 task.itinerary == id 的情况）
            try:
                sameJobOpts[int(getattr(job_obj, "idItinerary"))] = sorted_ops
            except Exception:
                pass
    # [SANITY] 订单内工序 idOperation 递增
    if os.environ.get("DEBUG_ASSERT", "1") == "1":
        for k, ops in sameJobOpts.items():
            last = -float("inf")
            for op in ops:
                assert op.idOperation > last, f"sameJob_op order broken for job={k}"
                last = op.idOperation
    return sameJobOpts


# ===================================================================
# VISUALIZATION FUNCTIONS (kept, but fix MultiDiGraph iteration)
# ===================================================================

def plot_graph(g, taskList, machineList, draw: bool = True,
               node_type_color_dict: dict = None,
               edge_type_color_dict: dict = None,
               half_width=None,
               half_height=None,
               **kwargs):
    import pylab
# matplotlib 在绘图函数中按需导入，避免训练期的额外依赖
    node_colors, _ = get_node_color_map(g, taskList, machineList, node_type_color_dict)
    edge_colors = get_edge_color_map(g, machineList, edge_type_color_dict)
    pos = calc_positions(g, taskList, machineList, half_width, half_height)

    if not kwargs:
        kwargs['figsize'] = (10, 5)
        kwargs['dpi'] = 300

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(1, 1, 1)

    nx.draw(g, pos,
            node_color=node_colors,
            edge_color=edge_colors,
            with_labels=True,
            font_size=4,
            node_size=50,
            width=0.5,
            arrowsize=5,
            label=False,
            ax=ax)
    if draw:
        pylab.title('The disjunctive graph of a instance', fontsize=9)
        plt.show()
    else:
        return fig, ax


def get_node_color_map(g, taskList, machineList, node_type_color_dict=None):
    if node_type_color_dict is None:
        node_type_color_dict = OrderedDict()

    disjunctive_tasks = disjunctive_ops(taskList, machineList)
    colors = {}
    colorSets = []
    node_colors = ['#e9e7ef', '#f9906f', '#44cef6', '#2add9c', '#fff143', '#cca4e3', '#725e80', '#3de1ad', '#ef7a82',
                   '#ae7000', '#057748', '#f47983', '#ca6924', '#b35c44', '#c0ebd7', '#db5a6b', '#3eede7', '#f9908f',
                   '#eacd76', '#bce672']
    i = 0
    for machine, tasks in disjunctive_tasks.items():
        i += 1
        colors[machine] = node_colors[i % len(node_colors)]

    for n in g.nodes:
        node = g.nodes[n]['keyword']
        if not node.completed:
            node_type_color_dict['uncompleted'] = node_colors[0]
            colorSets.append(node_type_color_dict['uncompleted'])
            continue
        mac = node.assignedMachine
        node_type_color_dict[mac] = colors.get(mac, node_colors[1])
        colorSets.append(node_type_color_dict[mac])
    return colorSets, colors


def get_edge_color_map(g, machinesList, edge_type_color_dict=None):
    if edge_type_color_dict is None:
        edge_type_color_dict = OrderedDict()
        edge_type_color_dict['CONJUNCTIVE'] = '#DCDCDC'
        edge_colors = ['#ff4777', '#f9906f', '#44cef6', '#2add9c', '#fff143', '#cca4e3', '#725e80', '#3de1ad',
                       '#ef7a82', '#ae7000', '#057748', '#f47983', '#ca6924', '#b35c44', '#e9e7ef', '#c0ebd7',
                       '#db5a6b', '#3eede7', '#f9908f', '#eacd76', '#bce672']
        i = 0
        for machine in machinesList:
            edge_type_color_dict[machine.name] = edge_colors[i % len(edge_colors)]
            i += 1

    colors = []
    for u, v, k, data in g.edges(keys=True, data=True):
        edge_type = data.get('type', 'UNKNOWN')
        if edge_type == 'CONJUNCTIVE':
            colors.append(edge_type_color_dict['CONJUNCTIVE'])
        elif edge_type == 'DISJUNCTIVE':
            machine_name = data.get('machine', 'UNKNOWN')
            colors.append(edge_type_color_dict.get(machine_name, '#000000'))
        else:
            colors.append('#999999')
    return colors


def calc_positions(g, taskList, machineList, half_width=None, half_height=None):
    pos_dict = OrderedDict()
    if half_width is None:
        half_width = 30
    if half_height is None:
        half_height = 10

    sameJobOpts = sameJob_op(taskList)
    sameJobOpts = SortedDict(sameJobOpts)

    i = 0
    for jobName, tasks in sameJobOpts.items():
        taskOfJob = tasks[0].idItinerary
        tasks.sort(key=lambda j: j.idOperation)
        taskNum = len(tasks)

        for j in range(taskNum):
            taskID = tasks[j].idOperation
            TaskName = f'O{taskOfJob}{taskID}'
            pos_dict[TaskName] = np.array((j, i))
        i += 1

    return pos_dict


def randonColor():
    color_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = '#' + ''.join(random.choice(color_list) for _ in range(6))
    return color


# =========================
# SANITY CHECKS
# =========================
def _sanity_check_graph(g: nx.MultiDiGraph):
    if os.environ.get("DEBUG_ASSERT", "1") != "1":
        return

    # 节点唯一 & keyword 存在
    seen = set()
    for n, d in g.nodes(data=True):
        assert n not in seen, "Duplicate node name"
        seen.add(n)
        assert 'keyword' in d and isinstance(d['keyword'], Operation), "Node missing keyword Operation"

    # 边字段与索引范围
    nodes_idx = {n: i for i, n in enumerate(g.nodes)}
    for u, v, k, data in g.edges(keys=True, data=True):
        assert u in nodes_idx and v in nodes_idx, "Edge endpoints not in node set"
        et = data.get('internal_type', None)
        assert et in (CONJUNCTIVE_TYPE, DISJUNCTIVE_TYPE), "Edge internal_type invalid"

        # 处理时间要有限
        pt = float(data.get('processing_time', 0.0))
        assert np.isfinite(pt), "processing_time not finite"

    # 工件内顺序再验
    # 从节点上拉回 Operation，再按 job 分组检查 idOperation 递增
    job_buckets = defaultdict(list)
    for n, d in g.nodes(data=True):
        op = d['keyword']
        job_buckets[getattr(op, "idItinerary")].append(op)
    for jid, ops in job_buckets.items():
        ops = sorted(ops, key=lambda x: x.idOperation)
        last = -float("inf")
        for op in ops:
            assert op.idOperation > last, "Job precedence order broken"
            last = op.idOperation
