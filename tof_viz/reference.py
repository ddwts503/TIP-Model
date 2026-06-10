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
    p1, p2, p3 = landmarks  # p1=左耳珠, p2=右耳珠, p3=鼻の付け根 を想定
    print(f"[reference] 基準3点(3D):\n  p1={p1.round(1)}\n  p2={p2.round(1)}"
          f"\n  p3={p3.round(1)}")

    # --- 解剖学的座標系を構築 ---
    # ez: 3点平面の法線 = 上下軸(superoinferior)。元データの+Yを上向きに合わせる
    ez = np.cross(p2 - p1, p3 - p1)
    ez = ez / np.linalg.norm(ez)
    if ez[1] < 0:
        ez = -ez
    # ex: 左右軸(mediolateral)= 耳珠p1→p2 を ez に直交化
    ex = p2 - p1
    ex = ex - (ex @ ez) * ez
    ex = ex / np.linalg.norm(ex)
    # ey: 前後軸(anteroposterior)= ez×ex。顔の前(カメラ側=元-Z)を+に
    ey = np.cross(ez, ex)
    if ey[2] > 0:           # +Z(奥)を向いていたら反転(前を+にする)
        ey = -ey
        ex = -ex            # 右手系を保つため ex も反転
    origin = landmarks.mean(axis=0)
    R = np.vstack([ex, ey, ez])             # 行: x'=左右, y'=前後, z'=上下

    aligned = (pc.xyz - origin) @ R.T       # (N,3) 解剖座標
    print("[reference] 解剖学的座標系:  x'=左右(矢状面の法線)  "
          "y'=前後(前頭/背中の面の法線)  z'=上下(横断面の法線)")
    print("  輪切り: --axis x → 矢状面スライス / --axis y → 前頭(背中)面スライス"
          " / --axis z → 横断面スライス")

    # --- 変換後点群を CSV 保存 ---
    if save_csv is None:
        save_csv = os.path.expanduser("~/Desktop/aligned_reference.csv")
    inten = pc.intensity if pc.intensity is not None else np.zeros(len(aligned))
    out = np.column_stack([aligned, inten])
    np.savetxt(save_csv, out, delimiter=",", header="x,y,z,intensity",
               comments="", fmt="%.3f")
    print(f"[reference] 解剖座標に合わせた点群を保存: {save_csv}")
    print("  この整列CSVを輪切り: --axis x=矢状面 / --axis y=前頭(背中)面 / "
          "--axis z=横断面")

    # --- プレビュー: 3つの解剖学的ビュー ---
    xp, yp, zp = aligned[:, 0], aligned[:, 1], aligned[:, 2]
    fig, (axC, axSag, axT) = plt.subplots(1, 3, figsize=(16, 5.5))
    # 正面(coronal view): 左右 x' × 上下 z', 色=前後 y'
    axC.scatter(xp, zp, c=yp, cmap="turbo", s=2)
    axC.set_aspect("equal", adjustable="box")
    axC.set_xlabel("x' left-right [mm]"); axC.set_ylabel("z' up-down [mm]")
    axC.set_title("FRONT view (coronal)\ncolor = front-back")
    # 側面(sagittal view): 前後 y' × 上下 z', 色=左右 x'
    axSag.scatter(yp, zp, c=xp, cmap="coolwarm", s=2)
    axSag.set_aspect("equal", adjustable="box")
    axSag.set_xlabel("y' front-back [mm]  (front +)")
    axSag.set_ylabel("z' up-down [mm]")
    axSag.set_title("SIDE view (sagittal)\ncolor = left-right")
    # 上面(transverse view): 左右 x' × 前後 y', 色=上下 z'
    axT.scatter(xp, yp, c=zp, cmap="viridis", s=2)
    axT.set_aspect("equal", adjustable="box")
    axT.set_xlabel("x' left-right [mm]"); axT.set_ylabel("y' front-back [mm]")
    axT.set_title("TOP view (transverse)\ncolor = up-down")
    for ax in (axC, axSag, axT):
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    fig.suptitle("Anatomical reference frame from 3 points  "
                 "(x'=L-R sagittal, y'=front-back coronal, z'=up-down transverse)",
                 fontsize=12)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"[saved] {save}")
    else:
        plt.show()
    return save_csv
