"""ToF 点群の 3D 可視化と輪切り(断面)表示.

- 3D 散布図(matplotlib / Open3D)
- 任意軸(x/y/z)に沿った輪切り: 単一断面 / 複数断面パネル / インタラクティブ slider
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .loader import PointCloud

_AXIS = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str) -> int:
    a = axis.lower()
    if a not in _AXIS:
        raise ValueError(f"axis は x/y/z のいずれか: {axis}")
    return _AXIS[a]


def _set_equal_aspect_3d(ax, xyz: np.ndarray) -> None:
    """3D 軸を等比(立方体)にする。"""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    span = span if span > 0 else 1.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def show_3d(
    pc: PointCloud,
    *,
    color_by: str = "z",
    point_size: float = 2.0,
    backend: str = "matplotlib",
    save: Optional[str] = None,
):
    """点群を 3D 表示する。

    color_by: "z"(高さ) / "intensity"(強度・あれば) / "mono"
    backend:  "matplotlib" / "open3d"
    """
    if backend == "open3d":
        return _show_3d_open3d(pc, color_by=color_by)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    c, cmap, label = _color_values(pc, color_by)
    sc = ax.scatter(pc.x, pc.y, pc.z, c=c, cmap=cmap, s=point_size,
                    depthshade=True)
    if c is not None and not np.isscalar(c):
        fig.colorbar(sc, ax=ax, shrink=0.6, label=label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"ToF 3D point cloud  (N={len(pc):,})")
    _set_equal_aspect_3d(ax, pc.xyz)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"[saved] {save}")
    else:
        plt.show()
    return fig


def _color_values(pc: PointCloud, color_by: str):
    if color_by == "intensity" and pc.intensity is not None:
        return pc.intensity, "viridis", "intensity"
    if color_by == "mono":
        return "steelblue", None, ""
    return pc.z, "viridis", "Z (height)"


def _show_3d_open3d(pc: PointCloud, *, color_by: str = "z"):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc.xyz)

    vals = pc.intensity if (color_by == "intensity" and pc.intensity is not None) else pc.z
    vmin, vmax = float(vals.min()), float(vals.max())
    norm = (vals - vmin) / (vmax - vmin + 1e-12)
    import matplotlib.cm as cm
    colors = cm.get_cmap("viridis")(norm)[:, :3]
    cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([cloud], window_name="ToF 3D (Open3D)")
    return cloud


def slice_points(
    pc: PointCloud,
    *,
    axis: str = "z",
    position: float,
    thickness: float,
) -> np.ndarray:
    """指定軸の position を中心に ±thickness/2 のスラブ内の点のブール mask を返す。"""
    ai = _axis_index(axis)
    coord = pc.xyz[:, ai]
    half = thickness / 2.0
    return np.abs(coord - position) <= half


def show_slice(
    pc: PointCloud,
    *,
    axis: str = "z",
    position: Optional[float] = None,
    thickness: Optional[float] = None,
    color_by: str = "z",
    point_size: float = 4.0,
    save: Optional[str] = None,
):
    """単一の輪切り(断面)を 2D 表示する。

    position 未指定なら指定軸の中央値、thickness 未指定なら全幅の 5%。
    """
    import matplotlib.pyplot as plt

    ai = _axis_index(axis)
    coord = pc.xyz[:, ai]
    if position is None:
        position = float(np.median(coord))
    if thickness is None:
        thickness = float((coord.max() - coord.min()) * 0.05) or 1.0

    mask = slice_points(pc, axis=axis, position=position, thickness=thickness)
    plane = [i for i in range(3) if i != ai]
    names = ["X", "Y", "Z"]

    fig, ax = plt.subplots(figsize=(8, 7))
    pts = pc.xyz[mask]
    if len(pts) == 0:
        ax.text(0.5, 0.5, "no points in this slice", ha="center",
                va="center", transform=ax.transAxes)
    else:
        c = pts[:, ai] if color_by != "intensity" or pc.intensity is None \
            else pc.intensity[mask]
        sc = ax.scatter(pts[:, plane[0]], pts[:, plane[1]], c=c,
                        cmap="viridis", s=point_size)
        fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_xlabel(names[plane[0]])
    ax.set_ylabel(names[plane[1]])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"slice  {axis.upper()} = {position:.3g} +/- {thickness/2:.3g}"
        f"   (n = {int(mask.sum()):,})"
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"[saved] {save}")
    else:
        plt.show()
    return fig


def show_slices_grid(
    pc: PointCloud,
    *,
    axis: str = "z",
    n: int = 9,
    thickness: Optional[float] = None,
    point_size: float = 3.0,
    save: Optional[str] = None,
):
    """指定軸に沿って等間隔に n 枚の断面をグリッド表示する。"""
    import math

    import matplotlib.pyplot as plt

    ai = _axis_index(axis)
    coord = pc.xyz[:, ai]
    lo, hi = float(coord.min()), float(coord.max())
    positions = np.linspace(lo, hi, n + 2)[1:-1]
    step = (hi - lo) / (n + 1)
    if thickness is None:
        thickness = step  # 断面が連続するように

    plane = [i for i in range(3) if i != ai]
    names = ["X", "Y", "Z"]

    # 先頭に「どこを切っているか」の案内図(locator)を1枚加える
    total = n + 1
    ncols = math.ceil(math.sqrt(total))
    nrows = math.ceil(total / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

    # 全断面で軸範囲を揃える
    xlim = (pc.xyz[:, plane[0]].min(), pc.xyz[:, plane[0]].max())
    ylim = (pc.xyz[:, plane[1]].min(), pc.xyz[:, plane[1]].max())

    # --- 案内図: 横=plane[0], 縦=切る軸, 色=plane[1] ---
    loc_ax = axes[0][0]
    loc_ax.scatter(pc.xyz[:, plane[0]], pc.xyz[:, ai],
                   c=pc.xyz[:, plane[1]], cmap="viridis", s=1)
    for k, pos in enumerate(positions, start=1):
        loc_ax.axhline(pos, color="red", lw=0.8)
        loc_ax.text(xlim[1], pos, f" {k}", color="red", fontsize=8,
                    va="center", ha="left")
    loc_ax.set_title("WHERE each slice is cut\n(red line = slice)",
                     fontsize=9)
    loc_ax.set_xlabel(names[plane[0]], fontsize=8)
    loc_ax.set_ylabel(f"{names[ai]} (slice axis)", fontsize=8)
    loc_ax.set_xlim(xlim)

    for k, pos in enumerate(positions):
        cell = k + 1  # 0番は案内図
        r, c = divmod(cell, ncols)
        ax = axes[r][c]
        mask = slice_points(pc, axis=axis, position=pos, thickness=thickness)
        pts = pc.xyz[mask]
        if len(pts):
            ax.scatter(pts[:, plane[0]], pts[:, plane[1]],
                       c=pts[:, ai], cmap="viridis", s=point_size)
        ax.set_title(f"#{k+1}  {axis.upper()}={pos:.3g} (n={int(mask.sum())})",
                     fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(names[plane[0]], fontsize=8)
        ax.set_ylabel(names[plane[1]], fontsize=8)

    # 余ったパネルを非表示
    for k in range(total, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"slices overview  axis={axis.upper()}  ({n} slices)  "
                 f"-- top-left = locator",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save:
        fig.savefig(save, dpi=150)
        print(f"[saved] {save}")
    else:
        plt.show()
    return fig


def interactive_slice(
    pc: PointCloud,
    *,
    axis: str = "z",
    thickness: Optional[float] = None,
    point_size: float = 4.0,
):
    """スライダーで断面位置を動かせるインタラクティブ表示(GUI 必須)。"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ai = _axis_index(axis)
    coord = pc.xyz[:, ai]
    lo, hi = float(coord.min()), float(coord.max())
    if thickness is None:
        thickness = (hi - lo) * 0.05 or 1.0

    plane = [i for i in range(3) if i != ai]
    names = ["X", "Y", "Z"]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.18)

    init = (lo + hi) / 2
    mask = slice_points(pc, axis=axis, position=init, thickness=thickness)
    pts = pc.xyz[mask]
    scat = ax.scatter(pts[:, plane[0]], pts[:, plane[1]],
                      c=pts[:, ai], cmap="viridis", s=point_size)
    ax.set_xlim(pc.xyz[:, plane[0]].min(), pc.xyz[:, plane[0]].max())
    ax.set_ylim(pc.xyz[:, plane[1]].min(), pc.xyz[:, plane[1]].max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(names[plane[0]])
    ax.set_ylabel(names[plane[1]])
    title = ax.set_title("")

    sax = plt.axes((0.15, 0.06, 0.7, 0.03))
    slider = Slider(sax, f"{axis.upper()} pos", lo, hi, valinit=init)

    def update(val):
        m = slice_points(pc, axis=axis, position=val, thickness=thickness)
        p = pc.xyz[m]
        scat.set_offsets(p[:, plane] if len(p) else np.empty((0, 2)))
        if len(p):
            scat.set_array(p[:, ai])
        title.set_text(
            f"slice {axis.upper()}={val:.3g} +/- {thickness/2:.3g}  (n={int(m.sum())})"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(init)
    plt.show()
    return slider
