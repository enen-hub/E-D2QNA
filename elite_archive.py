# -*- coding: utf-8 -*-
# utils/elite_archive.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import torch


def _dominates_min(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> bool:
    return np.all(a <= b + eps) and np.any(a < b - eps)


def _nds_filter_min(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if points.size == 0:
        return points
    keep = []
    for i, p in enumerate(points):
        dom = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates_min(q, p, eps):
                dom = True
                break
        if not dom:
            keep.append(i)
    return np.asarray(keep, dtype=int)


class EliteArchive:
    """
    维护非支配的精英轨迹集合：
    item = {
      'pref': np.ndarray(2,),        # 训练时使用的偏好向量
      'obj':  np.ndarray(2,),        # (cmax, ttotal)  —— 最小化
      'traj': List[transition]       # (state, action, next_state, done, logits/optional)
    }
    """

    def __init__(self, capacity: int = 50, quality_threshold: float = 0.7):
        self.capacity = int(capacity)
        self.items: List[Dict[str, Any]] = []
        # 质量阈值（用于中后期筛选），含语义但不作为绝对门槛（仍结合阶段判断）
        self.quality_threshold: float = float(quality_threshold)
        # 动态维护当前最佳目标（用于“支配当前最优”判断）
        self._best_obj: Optional[np.ndarray] = None
        # 增广配置：按概率对采样出的轨迹进行轻噪声变体生成
        self.augment_prob: float = 0.35
        self.augment_noise_scale: float = 0.05
        # 轨迹压缩配置：默认启用，保留30%关键转换并保留首尾
        self.compress_enabled: bool = True
        self.compression_ratio: float = 0.3
        self.preserve_endpoints: bool = True

    def add(self, pref, objectives, traj: List[Dict[str, Any]], epoch: Optional[int] = None):
        obj = np.array(objectives, dtype=float).reshape(2,)
        # 阶段化质量控制（若提供 epoch）：
        if epoch is not None:
            if not self._is_high_quality(obj, epoch):
                return  # 直接拒绝低质量样本，避免过拟合风险
        # 可选：在入档前进行轨迹压缩（保留高价值/大幅状态变化的关键步）
        try:
            if self.compress_enabled and isinstance(traj, list) and len(traj) > 0:
                traj = compress_trajectory(traj, compression_ratio=self.compression_ratio, keep_endpoints=self.preserve_endpoints)
        except Exception:
            pass
        # 记录并更新当前最优
        meta = {
            'pref': np.array(pref, dtype=float),
            'obj': obj,
            'traj': traj,
            'epoch': int(epoch) if epoch is not None else None,
            'diversity_score': self._compute_diversity(obj)
        }
        self.items.append(meta)
        self._update_best(obj)
        self._prune()

    def _prune(self):
        # 非支配过滤 + 截断容量
        objs = np.array([it['obj'] for it in self.items], dtype=float)
        idx = _nds_filter_min(objs)
        self.items = [self.items[i] for i in idx]
        if len(self.items) > self.capacity:
            # 多样性优先 + 适度考虑新旧（优先保留多样性高者；同分下保留较新样本）
            def _key(it):
                # 越大越好；所以按负值排序后截断前 capacity 个
                ds = float(it.get('diversity_score', 0.0))
                ep = it.get('epoch', -1)
                return (-ds, -ep)
            self.items.sort(key=_key)
            self.items = self.items[: self.capacity]

    # ======== 质量控制与多样性评估 ========
    def _update_best(self, obj: np.ndarray):
        try:
            if self._best_obj is None:
                self._best_obj = obj.copy()
            else:
                self._best_obj = np.minimum(self._best_obj, obj)
        except Exception:
            pass

    def _hv_contribution_proxy(self, obj: np.ndarray) -> float:
        """近似“超体积贡献”代理：使用相对改进量衡量。
        - 以当前最优 self._best_obj 为参考，计算规范化距离改进。
        - 返回值范围约为 [0, +inf)，典型建议阈值 ~ 0.01。
        """
        if self._best_obj is None:
            return 1.0  # 无参照时视为有贡献
        ref = np.maximum(self._best_obj, 1e-6)
        # 规范化距离：sum((ref - obj) / ref)_+
        diffs = np.maximum(ref - obj, 0.0) / ref
        return float(np.sum(diffs))

    def _dominates_current_best(self, obj: np.ndarray) -> bool:
        if self._best_obj is None:
            return True
        return _dominates_min(obj, self._best_obj)

    def _hypervolume_contribution(self, obj: np.ndarray) -> float:
        """2D超体积贡献的简化代理实现：
        - 若已有项为空，视为有贡献（返回正值）
        - 否则基于当前非支配集的近似贡献（复用 hv 代理方法）
        说明：真实HV需要参考点，这里使用当前最优为参照的贡献代理。
        """
        try:
            return float(self._hv_contribution_proxy(obj))
        except Exception:
            return 0.0

    def _is_high_quality(self, obj: np.ndarray, epoch: int) -> bool:
        """动态质量阈值（修复版）
        - 阶段1: 广泛收集 (0-500轮)，无条件接纳
        - 阶段2: 峰值期密集采样 (500-800轮)，只要不被严格支配就接纳
        - 阶段3: 选择性保存 (800轮以后)，要求对超体积有实际贡献 > 0.005
        """
        # 阶段1：0-500
        if epoch < 500:
            return True
        # 阶段2：500-800（不被现有任何解严格支配）
        elif epoch < 800:
            if not self.items:
                return True
            existing = np.array([it['obj'] for it in self.items], dtype=float)
            strictly_dominated = np.any(
                np.all(existing <= obj, axis=1) & np.any(existing < obj, axis=1)
            )
            return not strictly_dominated
        # 阶段3：>=800（要求超体积贡献）
        else:
            return self._hypervolume_contribution(obj) > 0.005

    def _compute_diversity(self, obj: np.ndarray) -> float:
        if not self.items:
            return 1.0
        center = np.mean([it['obj'] for it in self.items], axis=0)
        # 目标空间的 L2 距离作为多样性分数
        return float(np.linalg.norm(obj - center))

    def sample_trajectories(self, n: int) -> List[Dict[str, Any]]:
        if not self.items:
            return []
        n = int(min(n, len(self.items)))
        # 均匀随机采样轨迹
        idx = np.random.choice(len(self.items), size=n, replace=False)
        sampled = [self.items[i] for i in idx]
        # 采样期增广：按概率对轨迹添加轻微高斯噪声（仅作用于特征，不改变拓扑）
        out = []
        for it in sampled:
            try:
                if np.random.rand() < self.augment_prob:
                    aug_traj = augment_elite_trajectory(it['traj'], noise_scale=self.augment_noise_scale)
                    out.append({'pref': it['pref'], 'obj': it['obj'], 'traj': aug_traj})
                else:
                    out.append(it)
            except Exception:
                out.append(it)
        return out

    def __len__(self):
        return len(self.items)

    def to_list(self) -> List[Dict[str, Any]]:
        return self.items


# ======== 近邻变体增广 ========
def add_gaussian_noise(x, scale: float = 0.05):
    """对张量或数组添加高斯噪声；保持类型与设备。
    - 仅应用于特征向量；不改变图拓扑(edge_index)。
    """
    if torch.is_tensor(x):
        noise = torch.randn_like(x) * float(scale)
        return x + noise
    arr = np.asarray(x, dtype=np.float32)
    noise = np.random.randn(*arr.shape).astype(np.float32) * float(scale)
    return arr + noise


def augment_elite_trajectory(trajectory: List[Dict[str, Any]], noise_scale: float = 0.05) -> List[Dict[str, Any]]:
    """对精英轨迹添加轻微扰动，生成近邻变体。
    - 输入为包含 state={'features', 'edge_index'} 与 'action' 的列表
    - 仅对 state.features 加噪；edge_index 原样保留
    """
    augmented: List[Dict[str, Any]] = []
    for t in (trajectory or []):
        try:
            s = t.get('state', {})
            feats = s.get('features')
            aug_feats = add_gaussian_noise(feats, scale=noise_scale)
            augmented.append({
                **t,
                'state': {
                    **s,
                    'features': aug_feats
                }
            })
        except Exception:
            augmented.append(t)
    return augmented


# ======== 轨迹压缩（保留关键转换） ========
def _get_feat_tensor(step: Dict[str, Any]) -> Optional[torch.Tensor]:
    try:
        s = step.get('state', {})
        f = s.get('features', None)
        if f is None:
            return None
        return f if torch.is_tensor(f) else torch.tensor(np.asarray(f, dtype=np.float32))
    except Exception:
        return None


def _transition_score(curr: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> float:
    """为每个转换计算重要性分数：
    优先使用显式的 'td_error' 或 'advantage'；否则用相邻状态特征的 L2 变化幅度。
    """
    # 1) 显式信号
    for k in ('td_error', 'advantage'):
        v = curr.get(k, None)
        try:
            if v is not None:
                arr = np.asarray(v, dtype=np.float32)
                return float(np.abs(arr).mean())
        except Exception:
            pass
    # 2) 状态特征变化
    try:
        if prev is not None:
            f1 = _get_feat_tensor(prev)
            f2 = _get_feat_tensor(curr)
            if (f1 is not None) and (f2 is not None):
                # 保持在CPU计算以简化
                f1 = f1.detach().cpu().reshape(-1)
                f2 = f2.detach().cpu().reshape(-1)
                diff = f2 - f1
                return float(torch.norm(diff, p=2).item())
    except Exception:
        pass
    # 3) 回退：常数极小分数
    return 0.0


def compress_trajectory(trajectory: List[Dict[str, Any]], compression_ratio: float = 0.3, keep_endpoints: bool = True) -> List[Dict[str, Any]]:
    """保留轨迹中高价值的转换。
    - compression_ratio: 0~1，保留比例（至少保留1个，建议0.2~0.5）
    - keep_endpoints: 保留首尾转换以保留上下文与终局动作
    策略：
      1) 若存在 'td_error'/'advantage' 字段则以其绝对值排序；
      2) 否则使用相邻 state.features 的L2变化幅度作为重要性分数；
      3) 保证稳定性：始终保留最后一步；若启用 keep_endpoints 也保留第一步。
    """
    steps = trajectory or []
    n = len(steps)
    if n <= 2:
        return steps
    k = max(1, int(round(float(compression_ratio) * n)))
    # 计算分数
    scores = []
    prev = None
    for i, t in enumerate(steps):
        s = _transition_score(t, prev)
        scores.append((i, s))
        prev = t
    # 排除强制保留的端点
    forced = set()
    if keep_endpoints:
        forced.add(0)
    forced.add(n - 1)
    selectable = [(i, s) for (i, s) in scores if i not in forced]
    # 选择 top-k（扣除强制保留数）
    keep_budget = max(0, k - len(forced))
    selectable.sort(key=lambda x: x[1], reverse=True)
    selected = [i for (i, _) in selectable[:keep_budget]]
    selected.extend(list(forced))
    selected = sorted(set(selected))
    return [steps[i] for i in selected]