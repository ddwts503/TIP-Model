"""任意ライン(2点指定)に沿った斜め輪切り.

顔の正面図で2点(例: 左右の口角、耳珠と鼻の付け根)を指定すると、
その2点を通る縦の平面で切った断面プロフィール(ライン方向 u × 奥行き Z)を表示する。
2点はマウスクリックで指定するか、--p1/--p2 で座標指定できる。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .loader import PointCloud

Pt = Tuple[float, float]


def _profile_along_line(pc: PointCloud, p1: Pt, p2: Pt, thickness: float):
    """2点を通る縦平面の ±thickness/2 内の点を取り、ライン座標uと奥行きZを返す。"""
    xy = pc.xyz[:, :2]
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = p2 - p1
    L = float(np.hypot(*d))
    if L < 1e-6:
        raise ValueError("2点が近すぎます。離れた2点を指定してください。")
    dhat = d / L
    nhat = np.array([-dhat[1], dhat[0]])
    rel = xy - p1
    u = rel @ dhat            # ライン方向の位置
    w = rel @ nhat            # ラインからの垂直距離
    mask = np.abs(w) <= thickness / 2.0
    return u[mask], pc.xyz[mask, 2], mask, (p1, p2, dhat, nhat, L)


def oblique_slice(
    pc: PointCloud,
    *,
    p1: Optional[Pt] = None,
    p2: Optional[Pt] = None,
    thickness: float = 8.0,
    point_size: float = 6.0,
    save: Optional[str] = None,
):
    """2点を通るラインに沿った斜め断面を表示する。

    p1/p2 未指定なら、正面図上でマウスクリック2回で指定(GUI必須)。
    """
    import matplotlib.pyplot as plt

    # 正面図(X-Y, 色=奥行きZ)。手前ほど濃く見えるよう Z で色分け
    order = np.argsort(-pc.xyz[:, 2])  # 遠い点を先に描画→近い点が上
    fx, fy, fz = pc.xyz[order, 0], pc.xyz[order, 1], pc.xyz[order, 2]

    if p1 is None or p2 is None:
        # クリックで2点指定
        figp, axp = plt.subplots(figsize=(7, 7))
        axp.scatter(fx, fy, c=fz, cmap="turbo", s=2)
        axp.set_aspect("equal", adjustable="box")
        axp.set_xlabel("X [mm]"); axp.set_ylabel("Y [mm]")
        axp.set_title("Click 2 points (e.g. both mouth corners,\n"
                      "or tragus and nose root), then close is automatic")
        print("[oblique] 正面図で2点をクリックしてください...")
        pts = figp.ginput(2, timeout=0)
        plt.close(figp)
        if len(pts) < 2:
            raise ValueError("2点が取得できませんでした。もう一度お試しください。")
        p1, p2 = pts[0], pts[1]
        print(f"[oblique] 選択点: p1={p1[0]:.0f},{p1[1]:.0f}  "
              f"p2={p2[0]:.0f},{p2[1]:.0f}")

    u, z, mask, (a, b, dhat, nhat, L) = _profile_along_line(
        pc, p1, p2, thickness)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))

    # 左: 正面図 + 切断ライン
    axL.scatter(fx, fy, c=fz, cmap="turbo", s=2)
    axL.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=2)
    axL.scatter([a[0], b[0]], [a[1], b[1]], c="white", edgecolors="black",
                s=60, zorder=5)
    axL.annotate("p1", a, color="black", fontsize=10)
    axL.annotate("p2", b, color="black", fontsize=10)
    axL.set_aspect("equal", adjustable="box")
    axL.set_xlabel("X [mm]"); axL.set_ylabel("Y [mm]")
    axL.set_title("front view + cut line")

    # 右: 断面プロフィール(ライン方向 u × 奥行き Z)
    if len(u):
        axR.scatter(u, z, c=z, cmap="viridis", s=point_size)
    axR.set_xlabel("along line  u [mm]  (p1=0 -> p2)")
    axR.set_ylabel("depth Z [mm]  (small = closer)")
    axR.set_aspect("equal", adjustable="datalim")
    axR.invert_yaxis()  # 手前(小Z)を上に
    axR.set_title(f"oblique slice profile  (n={int(mask.sum())}, "
                  f"thickness={thickness:.0f}mm)")

    fig.suptitle("Oblique slice along a custom line", fontsize=13)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        from .loader import saved_and_open
        saved_and_open(save)
    else:
        plt.show()
    return p1, p2
