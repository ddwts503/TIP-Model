"""コマンドラインインターフェース.

使用例(CSV):
  python -m tof_viz data.csv --mode 3d
  python -m tof_viz data.csv --mode slice --axis z --pos 1.2 --thickness 0.1
  python -m tof_viz data.csv --mode grid --axis z --n 9 --save grid.png

使用例(ToForge .dat):
  python -m tof_viz sensor.dat --list-frames
  python -m tof_viz sensor.dat --frame 500 --mode 3d
  python -m tof_viz sensor.dat --frame 500 --mode grid --axis z --n 9
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


def _is_dat(path: str) -> bool:
    return path.lower().endswith(".dat")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tof_viz",
        description="ToF CSV データの 3D 可視化と輪切り(断面)表示",
    )
    p.add_argument("path", help="入力 CSV / テキストファイル")
    p.add_argument(
        "--mode",
        choices=["3d", "slice", "grid", "interactive", "browse", "contact"],
        default="3d",
        help="表示モード (default: 3d)。browse/contact は .dat のコマ探し用",
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
    # --- ToForge .dat 用 ---
    p.add_argument("--frame", type=int, default=None,
                   help="[.dat] 表示するフレーム番号(未指定なら中央のフレーム)")
    p.add_argument("--list-frames", action="store_true",
                   help="[.dat] フレーム数など情報だけ表示して終了")
    p.add_argument("--min-depth", type=float, default=500.0,
                   help="[.dat] 使用する最小距離 mm (default: 500)")
    p.add_argument("--max-depth", type=float, default=None,
                   help="[.dat] 使用する最大距離 mm(壁などを除外。例: 1500)")
    p.add_argument("--start", type=int, default=0,
                   help="[.dat contact] 一覧の開始フレーム")
    p.add_argument("--end", type=int, default=None,
                   help="[.dat contact] 一覧の終了フレーム")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if _is_dat(args.path):
        from . import toforge

        n = toforge.count_frames(args.path)
        if n == 0:
            print("このファイルは ToForge 形式として読めませんでした"
                  "(サイズが合いません)。", file=sys.stderr)
            return 1
        hdr = toforge.read_header(args.path, 0)
        print(f"[ToForge .dat] フレーム数: {n:,}  "
              f"(format {hdr['format_version']}, {toforge.SENSOR_WIDTH}x"
              f"{toforge.SENSOR_HEIGHT})")
        if args.list_frames:
            print(f"  --frame で 0〜{n-1} のフレームを選べます。"
                  f"例: --frame {n//2}")
            return 0

        if args.mode == "browse":
            from .browse import browse_frames
            browse_frames(args.path, start_frame=args.frame,
                          min_depth=args.min_depth, max_depth=args.max_depth)
            return 0

        if args.mode == "contact":
            from .browse import contact_sheet
            contact_sheet(args.path, n=args.n, start=args.start, end=args.end,
                          min_depth=args.min_depth, max_depth=args.max_depth,
                          save=args.save)
            return 0

        frame = args.frame if args.frame is not None else n // 2
        pc = toforge.read_frame(
            args.path, frame,
            min_depth=args.min_depth, max_depth=args.max_depth,
            max_points=args.max_points,
        )
    else:
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
