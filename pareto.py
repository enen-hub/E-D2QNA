# -*- coding: utf-8 -*-
import numpy as np

# === [PATCH B1] Pareto utilities for minimization (insert/replace) ===
def dominates_min(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> bool:
    """
    最小化问题的支配关系：a 支配 b 当且仅当：
      1) 对所有维度 a_i <= b_i + eps
      2) 且存在至少一维 a_i < b_i - eps
    """
    return np.all(a <= b + eps) and np.any(a < b - eps)


def non_dominated_filter_min(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    过滤非支配点（最小化）。输入/输出均为正值空间的目标。
    """
    if points.size == 0:
        return points
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and dominates_min(q, p, eps=eps):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points[np.array(keep, dtype=int)]


# === [PATCH B2] tolerant deduplication (insert) ===
def dedup_tolerant(points: np.ndarray, decimals=(0, 0)) -> np.ndarray:
    """
    浮点容忍去重：按每个维度的保留小数位进行四舍五入后唯一化。
    默认：二维(Cmax/Ttotal)按整数去重。
    """
    if points.size == 0:
        return points
    # 自适应维度：二维使用(0,0)，三维使用(0,0,2)
    if len(decimals) != points.shape[1]:
        decimals = tuple([0] * points.shape[1])
    rounded = np.stack([np.round(points[:, i], decimals=decimals[i]) for i in range(points.shape[1])], axis=1)
    uniq, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]
import matplotlib.pyplot as plt
import pandas as pd

# 可选：只有真的用到多样性指标时才需要 scipy
def _pdist_safe(X):
    try:
        from scipy.spatial.distance import pdist
        return pdist(X, metric='euclidean')
    except Exception:
        # 没装 scipy 时给个空数组，避免崩
        return np.array([])

# 兼容历史键名
def _key_map(k: str) -> str:
    alias = {'ltotal': 'ttotal', 'TTotal': 'ttotal', 'LTotal': 'ttotal'}
    return alias.get(k, k)

class ParetoAnalyzer:
    """可视化/导出 GA 的多目标结果：ga_trainer.pareto_history + history 字典"""

    def __init__(self, ga_trainer, history):
        self.ga = ga_trainer
        self.history = history
        self.pareto_history = getattr(ga_trainer, 'pareto_history', [])

    def plot_pareto_evolution_3d(self, epochs_to_plot=None, save_path=None):
        if not self.pareto_history:
            print("No Pareto history available")
            return
        # 仅二维显示（移除TEC）
        if epochs_to_plot is None:
            total = len(self.pareto_history)
            epochs_to_plot = [0, total // 2, total - 1]

        # 检查维度
        dims = None
        for ep in epochs_to_plot:
            if ep < len(self.pareto_history) and self.pareto_history[ep]:
                dims = np.array(self.pareto_history[ep]).shape[1]
                break
        dims = dims or 2

        fig = plt.figure(figsize=(15, 5))
        for i, ep in enumerate(epochs_to_plot):
            if ep >= len(self.pareto_history):
                continue
            pf = self.pareto_history[ep]
            if not pf:
                continue
            arr = np.array(pf, dtype=float)
            ax = fig.add_subplot(1, 3, i + 1)
            ax.scatter(arr[:, 0], arr[:, 1], s=30, alpha=0.8)
            ax.set_xlabel('Cmax'); ax.set_ylabel('Ttotal')
            ax.set_title(f'Epoch {ep} (PF size={len(pf)})')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_objective_trends(self, save_path=None):
        # 双目标：Cmax、Ttotal + HV
        makespan_hist = self.history.get('makespan', [])
        ttotal_hist = self.history.get('ttotal', self.history.get(_key_map('ltotal'), []))
        hv_hist = self.history.get('hypervolume', [])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(makespan_hist, linewidth=2)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cmax'); axes[0].set_title('Makespan')

        axes[1].plot(ttotal_hist, linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Ttotal'); axes[1].set_title('Total Tardiness')

        axes[2].plot(hv_hist, linewidth=2, color='purple')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Hypervolume'); axes[2].set_title('Hypervolume')

        for ax in axes.ravel(): ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pareto_diversity(self, save_path=None):
        if not self.pareto_history:
            print("No Pareto history available")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pf_sizes = [len(pf) for pf in self.pareto_history]
        axes[0].plot(pf_sizes, linewidth=2)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('PF Size'); axes[0].set_title('Pareto Front Size')
        axes[0].grid(True, alpha=0.3)

        # 简化的均匀性度量（需要 scipy 则尝试导入，否则跳过）
        spacing_values = []
        for pf in self.pareto_history:
            if len(pf) > 1:
                arr = np.array(pf, dtype=float)
                rng = np.ptp(arr, axis=0) + 1e-9
                norm = arr / rng
                d = _pdist_safe(norm)
                spacing_values.append(float(np.std(d)) if d.size else 0.0)
            else:
                spacing_values.append(0.0)
        axes[1].plot(spacing_values, linewidth=2, color='crimson')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Spacing (std dist)'); axes[1].set_title('Uniformity (lower better)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_objective_correlations(self, epoch=-1, save_path=None):
        try:
            import seaborn as sns
        except Exception:
            print("需要 seaborn 才能绘制相关性热图（pip install seaborn）")
            return
        if not self.pareto_history:
            print("No Pareto history available")
            return
        if epoch < 0:
            # 取最后一个非空 PF
            pf = []
            for cur in reversed(self.pareto_history):
                if cur:
                    pf = cur; break
        else:
            if epoch >= len(self.pareto_history) or not self.pareto_history[epoch]:
                print("Invalid epoch or empty PF")
                return
            pf = self.pareto_history[epoch]

        arr = np.array(pf, dtype=float)
        if arr.shape[0] < 2:
            print("Not enough solutions for correlation analysis")
            return
        # 二维列名（移除TEC）
        df = pd.DataFrame(arr[:, :2], columns=['Cmax', 'Ttotal'])
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'Objective Correlations (epoch={epoch if epoch >= 0 else "last"})')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def export_pareto_front(self, epoch=-1, filename="pareto_front.csv"):
        if not self.pareto_history:
            print("No Pareto history available")
            return
        # 取指定或最后一个非空
        pf = []
        if epoch < 0:
            for cur in reversed(self.pareto_history):
                if cur:
                    pf = cur; break
        else:
            if epoch < len(self.pareto_history):
                pf = self.pareto_history[epoch]
        if not pf:
            print("Empty Pareto front")
            return
        arr = np.array(pf, dtype=float)
        df = pd.DataFrame(arr[:, :2], columns=['Cmax', 'Ttotal'])
        df.to_csv(filename, index=False)
        print(f"Pareto front exported: {filename} ({len(df)} solutions)")
