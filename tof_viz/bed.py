"""ベッド(寝台)などの広い平面を自動検出し、それを基準(XY面)に整列する.

ベッド平面を Z=0 とし、ベッドからの高さ(手前=カメラ側)を +Z にする座標へ変換。
顔の3点クリックなしで、安定した基準面が得られる。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .loader import PointCloud
from .reference import _saved_and_open


def fit_bed_plane(xyz: np.ndarray, *, dist: float = 6.0, iters: int = 2000):
    """点群から支配的な平面(ベッド)を RANSAC で検出。

    Open3D があれば使用、無ければ簡易 RANSAC。戻り値: (n(3,), d, inlier_mask)
    平面: n·P + d = 0、|n|=1。
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        model, inliers = pcd.segment_plane(distance_threshold=dist,
                                           ransac_n=3, num_iterations=iters)
        a, b, c, d = model
        n = np.array([a, b, c], float)
        nn = np.linalg.norm(n)
        mask = np.zeros(len(xyz), bool)
        mask[np.asarray(inliers)] = True
        return n / nn, d / nn, mask
    except Exception:
        pass
    # 簡易 RANSAC フォールバック
    rng = np.random.default_rng(0)
    best_mask = None
    best_cnt = -1
    best = None
    N = len(xyz)
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p = xyz[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p[0]
        dist_all = np.abs(xyz @ n + d)
        mask = dist_all < dist
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask, best = cnt, mask, (n, d)
    n, d = best
    # インライアで最小二乗リフィット
    P = xyz[best_mask]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    d = -n @ c
    return n, d, best_mask


def _clean(xyz, inten, *, zlo_pct=1.0, zhi_pct=99.0):
    """遠近の外れ値ノイズを除去(Zのパーセンタイル + 統計的外れ値除去)。"""
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [zlo_pct, zhi_pct])
    m = (z >= lo) & (z <= hi)
    xyz, inten = xyz[m], inten[m]
    # Open3D 統計的外れ値除去(あれば)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        _, keep = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=2.0)
        keep = np.asarray(keep)
        mask = np.zeros(len(xyz), bool); mask[keep] = True
        xyz, inten = xyz[mask], inten[mask]
    except Exception:
        pass
    return xyz, inten


def align_to_bed(
    pc: PointCloud,
    *,
    dist: float = 6.0,
    save_csv: Optional[str] = None,
    save: Optional[str] = None,
):
    """ベッド平面を検出し、それを XY 面(Z=高さ)に整列した点群を保存・表示。"""
    import matplotlib.pyplot as plt

    inten0 = pc.intensity if pc.intensity is not None else np.zeros(len(pc.xyz))
    xyz, inten = _clean(pc.xyz, inten0)
    print(f"[bed] ノイズ除去: {len(pc.xyz):,} → {len(xyz):,} 点")

    n, d, mask = fit_bed_plane(xyz, dist=dist)
    # 法線をカメラ側(元-Z=手前)に向ける → 被写体の高さが +Z
    if n[2] > 0:
        n = -n
        d = -d
    # 高さ(ベッドからの符号付き距離)
    height = xyz @ n + d

    # 平面内の軸: 元X軸を平面に射影して x'、y'=zax×xax
    zax = n
    xax = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0, 0]) @ zax) * zax
    if np.linalg.norm(xax) < 1e-6:
        xax = np.array([0.0, 1.0, 0.0]) - (np.array([0, 1.0, 0]) @ zax) * zax
    xax = xax / np.linalg.norm(xax)
    yax = np.cross(zax, xax)

    origin = xyz.mean(0)
    origin = origin - (origin @ n + d) * n   # 重心をベッド面へ射影
    rel = xyz - origin
    xp = rel @ xax
    yp = rel @ yax
    zp = height                                # ベッドからの高さ
    aligned = np.column_stack([xp, yp, zp])

    if save_csv is None:
        save_csv = os.path.expanduser("~/Desktop/aligned_bed.csv")
    np.savetxt(save_csv, np.column_stack([aligned, inten]), delimiter=",",
               header="x,y,z,intensity", comments="", fmt="%.3f")
    print(f"[bed] ベッド平面を検出(インライア {int(mask.sum()):,} 点)。"
          f"法線={n.round(3)}")
    print(f"[bed] 高さの範囲 z' = [{zp.min():.0f}, {zp.max():.0f}] mm")
    print(f"[bed] ベッド=XY面(Z=高さ)に整列した点群を保存: {save_csv}")
    print(f"  Z'=ベッドからの高さmm(手前+)。輪切り --axis z はベッドに平行な層、"
          f"--axis x/y は垂直断面。")

    # 表示範囲はパーセンタイルで読みやすく
    def _lim(v, pad=0.05):
        lo, hi = np.percentile(v, [1, 99])
        m = (hi - lo) * pad + 1
        return lo - m, hi + m

    # プレビュー: 正面(x'-y' 色=高さ)と側面(x'-z')
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(13, 6))
    axF.scatter(xp, yp, c=np.clip(zp, *np.percentile(zp, [1, 99])),
                cmap="turbo", s=2)
    axF.set_aspect("equal", adjustable="box")
    axF.set_xlim(*_lim(xp)); axF.set_ylim(*_lim(yp))
    axF.set_xlabel("x' [mm]"); axF.set_ylabel("y' [mm]")
    axF.set_title("TOP view (looking down on bed)\ncolor = height above bed")
    axS.scatter(xp, zp, c=zp, cmap="turbo", s=2)
    axS.axhline(0, color="red", lw=1.2)        # ベッド面
    axS.set_xlim(*_lim(xp)); axS.set_ylim(*_lim(zp))
    axS.set_xlabel("x' [mm]"); axS.set_ylabel("z' height above bed [mm]")
    axS.set_title("SIDE view\nred line = bed (z'=0)")
    fig.suptitle("Aligned to BED plane (XY = bed, Z = height)", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        _saved_and_open(save)
    else:
        plt.show()
    return save_csv
