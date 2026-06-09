"""コマンドラインインターフェース.

使用例:
  python -m tof_viz data.csv --mode 3d
  python -m tof_viz data.csv --mode slice --axis z --pos 1.2 --thickness 0.1
  python -m tof_viz data.csv --mode grid --axis z --n 9 --save grid.png
  python -m tof_viz data.csv --mode interactive --axis z
"""
from __future__ import annotations

import argparse
import sys

from .loader import load_points
from .visualize import (
    interactive_slice,
    show_3d,
    show_slice,
    show_slices_grid,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tof_viz",
        description="ToF CSV データの 3D 可視化と輪切り(断面)表示",
    )
    p.add_argument("path", help="入力 CSV / テキストファイル")
    p.add_argument(
        "--mode",
        choices=["3d", "slice", "grid", "interactive"],
        default="3d",
        help="表示モード (default: 3d)",
    )
    p.add_argument("--axis", choices=["x", "y", "z"], default="z",
                   help="輪切りの軸 (default: z)")
    p.add_argument("--pos", type=float, default=None,
                   help="slice モードの断面位置(未指定なら中央値)")
    p.add_argument("--thickness", type=float, default=None,
                   help="断面の厚み(スラブ幅)")
    p.add_argument("--n", type=int, default=9,
                   help="grid モードの断面枚数 (default: 9)")
    p.add_argument("--color-by", choices=["z", "intensity", "mono"],
                   default="z", help="3d モードの色付け基準")
    p.add_argument("--backend", choices=["matplotlib", "open3d"],
                   default="matplotlib", help="3d モードの描画バックエンド")
    p.add_argument("--point-size", type=float, default=None,
                   help="点のサイズ")
    p.add_argument("--max-points", type=int, default=None,
                   help="読み込む最大点数(間引き)")
    p.add_argument("--delimiter", default=None,
                   help="区切り文字(未指定なら自動判定)")
    p.add_argument("--cols", default=None,
                   help="x,y,z[,intensity] の列番号をカンマ区切りで指定 (0始まり)")
    p.add_argument("--save", default=None,
                   help="画面表示せず画像ファイルに保存するパス")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cols = {}
    if args.cols:
        idx = [int(c) for c in args.cols.split(",")]
        names = ["x_col", "y_col", "z_col", "intensity_col"]
        cols = dict(zip(names, idx))

    pc = load_points(
        args.path,
        delimiter=args.delimiter,
        max_points=args.max_points,
        **cols,
    )
    print(f"[loaded] {len(pc):,} 点  from {pc.source}")
    print(f"  X: [{pc.x.min():.3g}, {pc.x.max():.3g}]"
          f"  Y: [{pc.y.min():.3g}, {pc.y.max():.3g}]"
          f"  Z: [{pc.z.min():.3g}, {pc.z.max():.3g}]")
    if pc.intensity is not None:
        print(f"  intensity: [{pc.intensity.min():.3g}, "
              f"{pc.intensity.max():.3g}]")

    if args.mode == "3d":
        ps = args.point_size or 2.0
        show_3d(pc, color_by=args.color_by, point_size=ps,
                backend=args.backend, save=args.save)
    elif args.mode == "slice":
        ps = args.point_size or 4.0
        show_slice(pc, axis=args.axis, position=args.pos,
                   thickness=args.thickness, color_by=args.color_by,
                   point_size=ps, save=args.save)
    elif args.mode == "grid":
        ps = args.point_size or 3.0
        show_slices_grid(pc, axis=args.axis, n=args.n,
                         thickness=args.thickness, point_size=ps,
                         save=args.save)
    elif args.mode == "interactive":
        if args.save:
            print("interactive モードは --save 非対応です。", file=sys.stderr)
        ps = args.point_size or 4.0
        interactive_slice(pc, axis=args.axis, thickness=args.thickness,
                          point_size=ps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
