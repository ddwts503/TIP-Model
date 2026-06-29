"""動作確認用のサンプル ToF 点群 CSV を生成する.

実物が無くてもツールを試せるよう、円柱+球+ノイズ床からなる
擬似 ToF 点群(x, y, z, intensity)を作って CSV 出力する。
"""
from __future__ import annotations

import argparse

import numpy as np


def make_sample(n: int = 30000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []

    # --- 円柱(縦に伸びた管) ---
    nc = n // 2
    theta = rng.uniform(0, 2 * np.pi, nc)
    r = 1.0 + rng.normal(0, 0.02, nc)
    zc = rng.uniform(0, 4.0, nc)
    cyl = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        zc,
        rng.uniform(0.6, 1.0, nc),  # intensity
    ])
    parts.append(cyl)

    # --- 上部の球(キャップ) ---
    ns = n // 4
    phi = rng.uniform(0, np.pi, ns)
    th = rng.uniform(0, 2 * np.pi, ns)
    rs = 1.2
    sph = np.column_stack([
        rs * np.sin(phi) * np.cos(th),
        rs * np.sin(phi) * np.sin(th),
        4.0 + rs * np.cos(phi),
        rng.uniform(0.3, 0.7, ns),
    ])
    parts.append(sph)

    # --- 床のノイズ平面 ---
    nf = n - nc - ns
    floor = np.column_stack([
        rng.uniform(-3, 3, nf),
        rng.uniform(-3, 3, nf),
        rng.normal(0, 0.03, nf),
        rng.uniform(0.0, 0.3, nf),
    ])
    parts.append(floor)

    pts = np.vstack(parts)
    rng.shuffle(pts)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description="サンプル ToF CSV 生成")
    ap.add_argument("-o", "--out", default="sample_tof.csv")
    ap.add_argument("-n", "--num", type=int, default=30000)
    args = ap.parse_args()

    pts = make_sample(args.num)
    header = "x,y,z,intensity"
    np.savetxt(args.out, pts, delimiter=",", header=header, comments="",
               fmt="%.5f")
    print(f"[written] {len(pts):,} 点 -> {args.out}")


if __name__ == "__main__":
    main()
