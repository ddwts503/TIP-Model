"""断面の長さ(弧長)と面積を計測し、前後(2データ)を比較する.

定義:
  長さ = 断面プロファイル曲線の弧長 [mm]
  面積 = その曲線と、両端を結ぶ弦で囲まれる面積 [mm^2]
基準座標に整列済みの点群(reference モードの出力CSV)に対して使う想定。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .loader import PointCloud

_AXIS = {"x": 0, "y": 1, "z": 2}


def measure_section(
    pc: PointCloud,
    *,
    axis: str = "z",
    position: float = 0.0,
    thickness: float = 5.0,
    nbins: int = 200,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """1つの断面の (長さmm, 面積mm^2, 曲線(N,2), 弦端点(2,2)) を返す。"""
    ai = _AXIS[axis.lower()]
    plane = [i for i in range(3) if i != ai]
    coord = pc.xyz[:, ai]
    mask = np.abs(coord - position) <= thickness / 2.0
    pts = pc.xyz[mask][:, plane]
    if len(pts) < 5:
        raise ValueError(f"断面に点が少なすぎます(n={len(pts)})。"
                         "thickness を大きく、または position を見直してください。")

    # 主成分(PCA)で輪郭の長手方向 t を求め、t に沿ってビン分け。
    # 各ビンの垂直方向 w の中央値で1本の輪郭曲線を作る(顔のような開いた緩い弧に最適)。
    mean = pts.mean(axis=0)
    P = pts - mean
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    e_long, e_perp = vt[0], vt[1]
    t = P @ e_long
    w = P @ e_perp
    edges = np.linspace(t.min(), t.max(), nbins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
    mt, mw = [], []
    for k in range(nbins):
        sel = idx == k
        if sel.sum() == 0:
            continue
        mt.append(t[sel].mean())
        mw.append(np.median(w[sel]))
    mt = np.asarray(mt)
    mw = np.asarray(mw)
    order = np.argsort(mt)
    mt, mw = mt[order], mw[order]
    # 2D座標に戻す
    curve = mean + np.outer(mt, e_long) + np.outer(mw, e_perp)
    if len(curve) < 2:
        raise ValueError("有効な曲線が作れませんでした。")

    # 弧長
    diffs = np.diff(curve, axis=0)
    length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

    # 面積: 曲線 + 端点を結ぶ弦で閉じた多角形の面積(シューレース)
    poly = np.vstack([curve, curve[0]])
    x, y = poly[:, 0], poly[:, 1]
    area = float(abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0)

    chord = np.array([curve[0], curve[-1]])
    return length, area, curve, chord


def _names():
    return ["X(L-R)", "Y(front-back)", "Z(up-down)"]


def measure_and_plot(
    pc: PointCloud,
    *,
    axis: str = "z",
    position: float = 0.0,
    thickness: float = 5.0,
    label: str = "data",
    compare_pc: Optional[PointCloud] = None,
    compare_label: str = "after",
    save: Optional[str] = None,
):
    """断面の長さ・面積を計測して表示。compare_pc があれば前後比較。"""
    import matplotlib.pyplot as plt

    ai = _AXIS[axis.lower()]
    plane = [i for i in range(3) if i != ai]
    names = _names()

    L1, A1, c1, chord1 = measure_section(pc, axis=axis, position=position,
                                         thickness=thickness)
    print(f"[measure] {label}:  長さ={L1:.1f} mm ({L1/10:.2f} cm)  "
          f"面積={A1:.0f} mm^2 ({A1/100:.2f} cm^2)  断面 {axis.upper()}={position}")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(c1[:, 0], c1[:, 1], "-o", ms=2, color="C0",
            label=f"{label}: L={L1/10:.2f}cm  A={A1/100:.2f}cm²")
    ax.fill(np.r_[c1[:, 0], c1[0, 0]], np.r_[c1[:, 1], c1[0, 1]],
            color="C0", alpha=0.15)

    result = {"label": label, "length_mm": L1, "area_mm2": A1}

    if compare_pc is not None:
        L2, A2, c2, _ = measure_section(compare_pc, axis=axis,
                                        position=position, thickness=thickness)
        print(f"[measure] {compare_label}:  長さ={L2:.1f} mm ({L2/10:.2f} cm)  "
              f"面積={A2:.0f} mm^2 ({A2/100:.2f} cm^2)")
        dL, dA = L2 - L1, A2 - A1
        print(f"[measure] 差分 ({compare_label} - {label}):  "
              f"Δ長さ={dL:+.1f} mm  Δ面積={dA:+.0f} mm^2 "
              f"({dA/100:+.2f} cm^2)")
        ax.plot(c2[:, 0], c2[:, 1], "-o", ms=2, color="C3",
                label=f"{compare_label}: L={L2/10:.2f}cm  A={A2/100:.2f}cm²")
        ax.fill(np.r_[c2[:, 0], c2[0, 0]], np.r_[c2[:, 1], c2[0, 1]],
                color="C3", alpha=0.15)
        ax.set_title(f"section {axis.upper()}={position}  "
                     f"ΔL={dL/10:+.2f}cm  ΔA={dA/100:+.2f}cm²")
        result["compare"] = {"label": compare_label, "length_mm": L2,
                             "area_mm2": A2, "dL_mm": dL, "dA_mm2": dA}
    else:
        ax.set_title(f"section {axis.upper()}={position}  "
                     f"L={L1/10:.2f}cm  A={A1/100:.2f}cm²")

    ax.set_xlabel(f"{names[plane[0]]} [mm]")
    ax.set_ylabel(f"{names[plane[1]]} [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        from .loader import saved_and_open
        saved_and_open(save)
    else:
        plt.show()
    return result
