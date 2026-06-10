"""3点で基準面を定義し、点群をその基準座標系に合わせ直す.

顔の正面図で3点(例: 左右の耳珠 + 鼻の付け根)をクリック、または座標指定すると、
その3点を通る平面を基準面とし:
  - 新しい原点 = 3点の重心
  - 新しい X 軸 = 1点目→2点目
  - 新しい Z 軸 = 基準面の法線(カメラ向きを正)
  - 新しい Y 軸 = Z×X
に座標変換した点群を作る。新Zは「基準面からの距離」になり、--axis z の輪切りが
基準面に平行な層になる。変換後の点群を CSV に保存し、通常の輪切り/3D表示に使える。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from .loader import PointCloud

Pt = Tuple[float, float]


def _nearest_xyz(pc: PointCloud, xy: Pt) -> np.ndarray:
    d2 = (pc.xyz[:, 0] - xy[0]) ** 2 + (pc.xyz[:, 1] - xy[1]) ** 2
    return pc.xyz[int(np.argmin(d2))].copy()


def define_reference(
    pc: PointCloud,
    *,
    pts: Optional[List[Pt]] = None,
    save_csv: Optional[str] = None,
    save: Optional[str] = None,
    point_size: float = 4.0,
) -> str:
    """3点で基準面を定義し、合わせ直した点群を CSV 保存。プレビューも表示/保存。"""
    import matplotlib.pyplot as plt

    order = np.argsort(-pc.xyz[:, 2])
    fx, fy, fz = pc.xyz[order, 0], pc.xyz[order, 1], pc.xyz[order, 2]

    if pts is None:
        figp, axp = plt.subplots(figsize=(7, 7))
        axp.scatter(fx, fy, c=fz, cmap="turbo", s=2)
        axp.set_aspect("equal", adjustable="box")
        axp.set_xlabel("X [mm]"); axp.set_ylabel("Y [mm]")
        axp.set_title("Click 3 reference points\n"
                      "(e.g. both tragus + nose root)")
        print("[reference] 基準面となる3点をクリックしてください...")
        clicks = figp.ginput(3, timeout=0)
        plt.close(figp)
        if len(clicks) < 3:
            raise ValueError("3点が取得できませんでした。もう一度お試しください。")
        pts = clicks

    landmarks = np.array([_nearest_xyz(pc, (p[0], p[1])) for p in pts])
    p1, p2, p3 = landmarks
    print(f"[reference] 基準3点(3D):\n  p1={p1.round(1)}\n  p2={p2.round(1)}"
          f"\n  p3={p3.round(1)}")

    origin = landmarks.mean(axis=0)
    xax = p2 - p1
    xax = xax / np.linalg.norm(xax)
    nrm = np.cross(p2 - p1, p3 - p1)
    nrm = nrm / np.linalg.norm(nrm)
    if nrm[2] > 0:           # 法線をカメラ向き(奥行きが手前向き)に
        nrm = -nrm
    yax = np.cross(nrm, xax)
    R = np.vstack([xax, yax, nrm])          # 各行が新軸

    aligned = (pc.xyz - origin) @ R.T       # (N,3) 新座標

    # --- 変換後点群を CSV 保存 ---
    if save_csv is None:
        save_csv = os.path.expanduser("~/Desktop/aligned_reference.csv")
    inten = pc.intensity if pc.intensity is not None else np.zeros(len(aligned))
    out = np.column_stack([aligned, inten])
    np.savetxt(save_csv, out, delimiter=",", header="x,y,z,intensity",
               comments="", fmt="%.3f")
    print(f"[reference] 基準座標に合わせた点群を保存: {save_csv}")
    print(f"  新Z = 基準面からの距離(手前が＋)。"
          f"これを輪切りすると基準面に平行な層になります。")

    # --- プレビュー: 正面(x'-y')と側面(x'-z') ---
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(13, 6))
    axF.scatter(aligned[:, 0], aligned[:, 1], c=aligned[:, 2],
                cmap="turbo", s=2)
    axF.axhline(0, color="k", lw=0.6); axF.axvline(0, color="k", lw=0.6)
    axF.set_aspect("equal", adjustable="box")
    axF.set_xlabel("x' [mm]"); axF.set_ylabel("y' [mm]")
    axF.set_title("front (aligned)  color = distance from plane")
    axS.scatter(aligned[:, 0], aligned[:, 2], c=aligned[:, 2],
                cmap="viridis", s=2)
    axS.axhline(0, color="red", lw=1.0)   # 基準面 z'=0
    axS.set_aspect("equal", adjustable="datalim")
    axS.set_xlabel("x' [mm]"); axS.set_ylabel("z' = dist from plane [mm]")
    axS.set_title("side (aligned)  red line = reference plane")
    fig.suptitle("Reference plane defined by 3 points", fontsize=13)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"[saved] {save}")
    else:
        plt.show()
    return save_csv
