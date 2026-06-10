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


def _saved_and_open(path: str) -> None:
    """保存を報告し、存在確認のうえ macOS では自動で開く。"""
    import os
    import subprocess
    import sys

    exists = os.path.exists(path)
    ap = os.path.abspath(path)
    print(f"[saved] {ap}  (exists={exists})")
    if exists and sys.platform == "darwin":
        try:
            subprocess.run(["open", ap], check=False)
        except Exception:
            pass


_LABELS = ["1: LEFT ear / tragus (left edge)",
           "2: RIGHT ear / tragus (right edge)",
           "3: nose root (center)"]


def save_front_map(pc: PointCloud, save: str, *, grid: float = 50.0,
                   binsize: float = 3.0):
    """顔の正面図を「塗りつぶしヒートマップ」で座標グリッド付き保存(くっきり)。

    点の散布ではなく X-Y を細かいマスに区切り各マスの奥行きで塗るので、
    顔の凹凸がはっきり見え、左耳/右耳/鼻根の X,Y を読み取りやすい。
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    x, y, z = pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    nx = max(10, int((xmax - xmin) / binsize))
    ny = max(10, int((ymax - ymin) / binsize))
    # 各マスの平均奥行き(なめらかな塗り)
    sumz, xe, ye = np.histogram2d(x, y, bins=[nx, ny],
                                  range=[[xmin, xmax], [ymin, ymax]],
                                  weights=z)
    cnt, _, _ = np.histogram2d(x, y, bins=[nx, ny],
                               range=[[xmin, xmax], [ymin, ymax]])
    with np.errstate(invalid="ignore"):
        grid_z = np.where(cnt > 0, sumz / cnt, np.nan)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(grid_z.T, origin="lower", cmap=cmap,
                   extent=[xmin, xmax, ymin, ymax], aspect="equal",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="depth Z [mm]")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(grid))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(grid))
    ax.grid(True, which="major", color="w", alpha=0.4, lw=0.5)
    ax.set_xlabel("X (left-right) [mm]")
    ax.set_ylabel("Y (up-down) [mm]")
    ax.set_title("FRONT map — read X,Y of: left ear / right ear / nose root\n"
                 "then run: --mode reference --p1=X,Y --p2=X,Y --p3=X,Y")
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    print("[front] 座標つき正面図(ヒートマップ):")
    _saved_and_open(save)
    print("  この画像で 左耳/右耳/鼻根 の X,Y を読み、--p1=X,Y --p2=X,Y --p3=X,Y "
          "に入れてください(マイナスは = でつなぐ)。")


def save_front_html(pc: PointCloud, save: str, *, max_points: int = 50000):
    """ブラウザで開く対話的な正面図(HTML)。点にマウスを当てると X,Y,Z 表示。

    matplotlib の窓が動かない環境向け。Chrome等で開き、ズーム・ホバーで
    耳/鼻根の座標を正確に読める。
    """
    import plotly.graph_objects as go

    xyz = pc.xyz
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[sel]
    fig = go.Figure(go.Scattergl(
        x=xyz[:, 0], y=xyz[:, 1], mode="markers",
        marker=dict(size=3, color=xyz[:, 2], colorscale="Turbo",
                    colorbar=dict(title="Z mm"), showscale=True),
        hovertemplate="X=%{x:.0f} mm<br>Y=%{y:.0f} mm<br>Z=%{marker.color:.0f} mm"
                      "<extra></extra>",
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="FRONT map (interactive) — hover to read X,Y of "
              "left ear / right ear / nose root",
        xaxis_title="X left-right [mm]", yaxis_title="Y up-down [mm]",
        width=900, height=850,
    )
    fig.write_html(save, include_plotlyjs="cdn")
    print("[front-html] ブラウザ用の操作画面(ホバーで座標表示):")
    _saved_and_open(save)


def _pick_3_points(fx, fy, fz):
    """顔の正面図で3点を1つずつ案内付きクリック。押した所に番号印を表示。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.scatter(fx, fy, c=fz, cmap="turbo", s=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("Set reference plane — Click  " + _LABELS[0],
                 fontsize=11)
    clicks: list = []

    def onclick(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        clicks.append((float(ev.xdata), float(ev.ydata)))
        ax.plot(ev.xdata, ev.ydata, "k+", ms=16, mew=2.5)
        ax.annotate(str(len(clicks)), (ev.xdata, ev.ydata),
                    color="black", fontsize=14, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points")
        if len(clicks) < 3:
            ax.set_title("Set reference plane — Click  "
                         + _LABELS[len(clicks)], fontsize=11)
        else:
            ax.set_title("3 points set — window closes automatically",
                         fontsize=11)
        fig.canvas.draw_idle()
        if len(clicks) >= 3:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    print("[reference] 顔の左の耳→右の耳→鼻の付け根 の順に3点クリックしてください。")
    plt.show()
    if len(clicks) < 3:
        raise ValueError("3点がクリックされませんでした。もう一度お試しください。")
    return clicks


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
        pts = _pick_3_points(fx, fy, fz)

    landmarks = np.array([_nearest_xyz(pc, (p[0], p[1])) for p in pts])
    p1, p2, p3 = landmarks  # p1=左耳珠, p2=右耳珠, p3=鼻の付け根 を想定
    print(f"[reference] 基準3点(3D):\n  p1={p1.round(1)}\n  p2={p2.round(1)}"
          f"\n  p3={p3.round(1)}")

    # --- 3点の妥当性チェック(近すぎ/一直線を防ぐ) ---
    d12 = np.linalg.norm(p2 - p1)
    d13 = np.linalg.norm(p3 - p1)
    d23 = np.linalg.norm(p3 - p2)
    if min(d12, d13, d23) < 15.0:
        raise ValueError(
            "3点が近すぎます(離れた3点を選び直してください)。"
            f" 点間距離= {d12:.0f},{d13:.0f},{d23:.0f} mm")
    cross = np.cross(p2 - p1, p3 - p1)
    if np.linalg.norm(cross) < 1e-3 * d12 * d13:
        raise ValueError("3点がほぼ一直線です。三角形になる3点を選び直してください。")

    # --- 解剖学的座標系を構築 ---
    # ez: 3点平面の法線 = 上下軸(superoinferior)。元データの+Yを上向きに合わせる
    ez = cross / np.linalg.norm(cross)
    if ez[1] < 0:
        ez = -ez
    # ex: 左右軸(mediolateral)= 耳珠p1→p2 を ez に直交化
    ex = p2 - p1
    ex = ex - (ex @ ez) * ez
    nx = np.linalg.norm(ex)
    if nx < 1e-6:
        raise ValueError("基準軸が作れません。3点を選び直してください。")
    ex = ex / nx
    # ey: 前後軸(anteroposterior)= ez×ex。顔の前(カメラ側=元-Z)を+に
    ey = np.cross(ez, ex)
    if ey[2] > 0:           # +Z(奥)を向いていたら反転(前を+にする)
        ey = -ey
        ex = -ex            # 右手系を保つため ex も反転
    origin = landmarks.mean(axis=0)
    R = np.vstack([ex, ey, ez])             # 行: x'=左右, y'=前後, z'=上下

    with np.errstate(all="ignore"):         # macOS BLAS の空振り警告を抑制
        aligned = (pc.xyz - origin) @ R.T   # (N,3) 解剖座標
    inten_all = pc.intensity if pc.intensity is not None else np.zeros(len(aligned))
    finite = np.isfinite(aligned).all(axis=1)
    if not finite.all():
        aligned = aligned[finite]
        inten_all = inten_all[finite]
    print("[reference] 解剖学的座標系:  x'=左右(矢状面の法線)  "
          "y'=前後(前頭/背中の面の法線)  z'=上下(横断面の法線)")
    print("  輪切り: --axis x → 矢状面スライス / --axis y → 前頭(背中)面スライス"
          " / --axis z → 横断面スライス")

    # --- 変換後点群を CSV 保存 ---
    if save_csv is None:
        save_csv = os.path.expanduser("~/Desktop/aligned_reference.csv")
    out = np.column_stack([aligned, inten_all])
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
        _saved_and_open(save)
    else:
        plt.show()
    return save_csv
