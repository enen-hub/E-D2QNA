import os
import sys
import re
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import argparse
from pathlib import Path
from statistics import mean

import numpy as np


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.cOperation import Operation
    from core.clMachine import Machine
    from core.clItinerary import Itinerary
except ImportError:
    print("ERROR: Could not import core classes. Make sure this script is in a subdirectory of your project root.")
    # 定义临时的占位符类，以允许文件独立运行
    class Operation: pass
    class Machine: pass
    class Itinerary: pass



def _get_itinerary_id(raw_op: dict, fallback_index: int = -1) -> Optional[int]:
    """统一的、健壮的工件ID解析器"""
    itinerary_id = raw_op.get("itinerary_id")
    if itinerary_id is not None:
        try: return int(itinerary_id)
        except (ValueError, TypeError): pass
    itinerary_id = raw_op.get("idItinerary")
    if itinerary_id is not None:
        try: return int(itinerary_id)
        except (ValueError, TypeError): pass
    it_name = raw_op.get("itinerary")
    if isinstance(it_name, str):
        digits = re.findall(r"\d+", it_name)
        if digits: return int(digits[0])
    if fallback_index != -1: return fallback_index
    return None

def _get_operation_id(raw_op: dict, fallback_index: Optional[int] = None) -> Optional[int]:
    
    for key in ("idOperation", "operation_id", "op_id", "idOp", "id_operation"):
        val = raw_op.get(key)
        if val is not None:
            try: return int(val)
            except (ValueError, TypeError): continue
    name = raw_op.get("operation") or raw_op.get("name")
    if isinstance(name, str):
        digits = re.findall(r"\d+", name)
        if digits:
            try: return int(digits[0])
            except ValueError: pass
    return fallback_index

# ------------------------------
# 单文件读取与验证（核心逻辑）
# ------------------------------
def _read_single_fjs_file(file_path: str, validate_due_dates: bool = False) -> Tuple[Optional[Dict], Optional[List[Itinerary]], Optional[List[Machine]]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # ---------- 1) 机器 ----------
        machines: List[Machine] = []
        machine_names = set()
        for raw_mac in raw_data.get("machines", []):
            mac_name = raw_mac.get("name")
            m = Machine(
                aName=mac_name,
                time=0.0,
            )
            machines.append(m)
            machine_names.add(mac_name)

        # ---------- 2) 将 initial_jobs 分组到各工件 ----------
        ops_by_job: Dict[str, List[dict]] = defaultdict(list)
        for idx, raw_op in enumerate(raw_data.get("initial_jobs", [])):
            itinerary_id = _get_itinerary_id(raw_op, fallback_index=idx)
            if itinerary_id is None: continue
            jid = str(itinerary_id)
            ops_by_job[jid].append(raw_op)

        # ---------- 3) 交期来源（工件级） ----------
        # 3.1 优先读取 JSON.jobs[].due_date
        json_due_map: Dict[str, int] = {}
        for j in raw_data.get("jobs", []):
            try:
                jid_raw = (j.get("itineraryName") or j.get("job_id") or j.get("idItinerary") or j.get("itinerary"))
                if jid_raw is None: continue
                digits = re.findall(r"\d+", str(jid_raw))
                if not digits: continue
                jid = str(int(digits[0]))
                for k in ("due_date", "dueDate", "duedate"):
                    if k in j and j[k] is not None:
                        json_due_map[jid] = int(j[k]); break
            except Exception: pass

        # 3.2 计算每个 job 的 LB
        lb_map: Dict[str, float] = {}
        for jid, ops in ops_by_job.items():
            lb = 0.0
            for raw_op in ops:
                raw_cm = raw_op.get("candidate_machines", raw_op.get("machine", {}))
                if isinstance(raw_cm, dict) and raw_cm:
                    try: mins = min(float(v) for v in raw_cm.values()); lb += mins
                    except Exception: pass
            lb_map[jid] = max(1.0, lb)

        # 3.3 参数
        meta_slack = raw_data.get("meta", {}).get("due_slack_range", [None, None])
        due_slack_min = float(os.environ.get("DUE_SLACK_MIN", meta_slack[0] or "1.1"))
        due_slack_max = float(os.environ.get("DUE_SLACK_MAX", meta_slack[1] or "1.8"))
        if due_slack_max < due_slack_min: due_slack_max = due_slack_min
        tardiness_factor = float(os.environ.get("TARDINESS_FACTOR", "0.5"))
        mirror_due_to_ops = os.environ.get("MIRROR_DUE_TO_OPS", "0") == "1"

        # 3.4 最终交期映射 + 统计
        job_metadata: Dict[str, Dict] = {}
        for jid_str in ops_by_job.keys():
            jid = int(jid_str)
            if jid_str in json_due_map:
                due = int(json_due_map[jid_str])
                source = "json"
            elif jid_str in lb_map and lb_map[jid_str] > 0:
                # 为了可复现验证，使用基于seed的随机数
                np.random.seed(int(jid))
                alpha = float(np.random.uniform(due_slack_min, due_slack_max))
                due = int(round(alpha * lb_map[jid_str]))
                source = "alpha_lb"
            else:
                avg_sum = 0.0
                for raw_op in ops_by_job[jid_str]:
                    cm = raw_op.get("candidate_machines", raw_op.get("machine", {}))
                    if isinstance(cm, dict) and cm:
                        vals = [float(v) for v in cm.values() if isinstance(v, (int, float))]
                        if vals: avg_sum += float(np.mean(vals))
                due = int(max(1.0, tardiness_factor * avg_sum))
                source = "legacy"
            job_metadata[jid_str] = {"due_date": int(max(1, due)), "_due_source": source, "_lb": float(lb_map.get(jid_str, 0.0))}

        # ---------- 4) 构造 Itinerary / Operation ----------
        final_jobs: List[Itinerary] = []
        for jid_str, ops_list in ops_by_job.items():
            meta = job_metadata[jid_str]
            job = Itinerary(jid_str, meta["due_date"])
            ops_with_ids = sorted([(_get_operation_id(ro, i), ro) for i, ro in enumerate(ops_list, 1)], key=lambda t: t[0])
            for op_id, raw_op in ops_with_ids:
                raw_cm = raw_op.get("candidate_machines", raw_op.get("machine", {}))
                candidate_machines = {m: float(d) for m, d in raw_cm.items() if m in machine_names and isinstance(d, (int, float))}
                if not candidate_machines: continue
                new_op = Operation(parent_job=job, operation_id=op_id, candidate_machines=candidate_machines)
                if mirror_due_to_ops: new_op.due_date = job.due_date
                job.operations.append(new_op)
            if job.operations: final_jobs.append(job)

        # ---------- 5) 诊断与验证信息 ----------
        due_stat = {"json": 0, "alpha_lb": 0, "legacy": 0}
        for v in job_metadata.values(): due_stat[v["_due_source"]] += 1
        
        print(f"SUCCESS: {os.path.basename(file_path)} | 工件: {len(final_jobs)} | 机器: {len(machines)}")
        
        raw_info = {"metadata": {"due_source_stat": due_stat, "mirror_due_to_ops": mirror_due_to_ops}}
        if 'meta' in raw_data and 'due_slack_range' in raw_data['meta']:
            raw_info['metadata']['original_slack_range'] = raw_data['meta']['due_slack_range']

        
        if validate_due_dates:
            _run_due_date_validation(file_path, job_metadata, json_due_map, final_jobs, [due_slack_min, due_slack_max])

        return raw_info, final_jobs, machines

    except Exception as e:
        print(f"ERROR: 数据加载失败: {file_path} -> {e}")
        import traceback; traceback.print_exc()
        return None, None, None

def _run_due_date_validation(file_path, job_metadata, json_due_map, final_jobs, slack_range):
    print(f"\n=== DUE DATE VALIDATION: {os.path.basename(file_path)} ===")
    print(f"α-range expectation: [{slack_range[0]:.3f}, {slack_range[1]:.3f}] (from meta or env)")
    
    headers = ["Job", "LB", "JSON_due", "Loaded_due", "alpha", "match_load", "alpha_in", "round_ok", "RESULT"]
    print(f" {' | '.join(h.center(10) for h in headers)}")
    print("-" * (len(headers) * 12))
    
    passed_count = 0
    job_map = {str(j.idItinerary): j for j in final_jobs}
    
    for jid in sorted(job_metadata.keys(), key=int):
        meta = job_metadata[jid]
        lb = meta["_lb"]
        json_due = json_due_map.get(jid, "N/A")
        loaded_job = job_map.get(jid)
        
        if not loaded_job:
            print(f" {jid.center(3)} | {'-'.center(8)} | {'-'.center(8)} | {'MISSING'.center(10)} | ... -> FAIL")
            continue
            
        loaded_due = loaded_job.due_date
        
        alpha = loaded_due / lb if lb > 0 else 0.0
        match_load = (str(json_due) == str(loaded_due)) if json_due != "N/A" else "N/A"
        alpha_in_range = (slack_range[0] <= alpha <= slack_range[1])
        
        
        np.random.seed(int(jid)) # 确保随机数一致
        recalc_alpha = np.random.uniform(slack_range[0], slack_range[1])
        recalc_due = int(round(recalc_alpha * lb)) if lb > 0 else 0
        round_ok = (loaded_due == recalc_due) or (meta["_due_source"] == "json")
        
        result = "OK" if (match_load in [True, "N/A"] and round_ok) else "FAIL"
        if result == "OK": passed_count += 1
            
        print(f" {jid.center(3)} | {lb:8.2f} | {str(json_due).center(8)} | {str(loaded_due).center(10)} | "
              f"{alpha:8.3f} | {str(match_load).center(10)} | {str(alpha_in_range).center(8)} | "
              f"{str(round_ok).center(8)} | {result}")
              
    print(f"\nValidation Result: {passed_count}/{len(job_metadata)} jobs passed.")
    if passed_count == len(job_metadata):
        print("[SUCCESS] Data loading is correct and consistent with generation logic.")
    else:
        print("[FAIL] Mismatches detected. Review data loading or generation logic.")


def readData(
    data_source: str, is_batch: bool = False, max_samples: Optional[int] = None,
    validate: bool = False 
) -> Union[Tuple, Tuple[List, List, List]]:
    """读取FJS调度数据，并可选择性地进行交叉验证"""
    if not os.path.exists(data_source): raise FileNotFoundError(f"数据源不存在: {data_source}")
    if is_batch:
        if not os.path.isdir(data_source): raise NotADirectoryError(f"批量读取需要目录: {data_source}")
        json_pattern = re.compile(r"dfjss_\d+[x×]\d+_seed\d+\.json", re.IGNORECASE)
        json_files = [os.path.join(r, f) for r, _, fs in os.walk(data_source) for f in fs if json_pattern.match(f)]
        if not json_files: raise ValueError(f"目录 {data_source} 中未找到符合命名规则的 JSON 文件")
        json_files.sort(key=lambda x: int(re.search(r"seed(\d+)\.json", x).group(1)))
        if max_samples: json_files = json_files[:max_samples]
        print(f"BATCH: 批量读取: {len(json_files)} 个文件")
        results = [_read_single_fjs_file(f, validate) for f in json_files]
        all_raw_info, all_jobs, all_machines = zip(*[r for r in results if r[0] is not None])
        print(f"SUCCESS: 批量读取完成: {len(all_raw_info)} 个样本")
        return list(all_raw_info), list(all_jobs), list(all_machines)
    if not os.path.isfile(data_source) or not data_source.endswith(".json"): raise ValueError(f"需要 JSON 文件: {data_source}")
    print(f"FILE: 单文件读取: {os.path.basename(data_source)}")
    return _read_single_fjs_file(data_source, validate)


def main_cli():
    parser = argparse.ArgumentParser(description="DFJSS Data Loader and Validator")
    parser.add_argument("path", help="Path to a single JSON file or a directory of datasets.")
    parser.add_argument("--batch", action="store_true", help="Enable batch mode to read all files in a directory.")
    parser.add_argument("--max", type=int, default=None, help="In batch mode, limit the number of files to read.")
    parser.add_argument("--validate", action="store_true", help="Run a detailed cross-validation check on due dates.")
    args = parser.parse_args()
    
    readData(args.path, is_batch=args.batch, max_samples=args.max, validate=args.validate)

if __name__ == "__main__":
    main_cli()