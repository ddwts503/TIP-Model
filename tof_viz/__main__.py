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
import glob
import os
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


def resolve_path(path: str) -> str:
    """フォルダが渡されたら中の .dat を再帰的に探して返す。

    ファイル/フォルダ名の日本語(NFC/NFD)差にも強い。複数あれば最大サイズを採用。
    """
    path = path.rstrip("/")
    if os.path.isdir(path):
        dats = glob.glob(os.path.join(path, "**", "*.dat"), recursive=True)
        dats = [d for d in dats if os.path.isfile(d)]
        if not dats:
            raise FileNotFoundError(
                f"フォルダ内に .dat が見つかりません: {path}"
            )
        dats.sort(key=lambda d: os.path.getsize(d), reverse=True)
        chosen = dats[0]
        if len(dats) > 1:
            print(f"[info] {len(dats)} 個の .dat が見つかりました。"
                  f"最大のものを使用します:")
        print(f"[info] 使用ファイル: {chosen}")
        return chosen
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tof_viz",
        description="ToF CSV データの 3D 可視化と輪切り(断面)表示",
    )
    p.add_argument("path", help="入力 CSV / テキストファイル")
    p.add_argument(
        "--mode",
        choices=["3d", "slice", "grid", "interactive", "browse", "contact",
                 "oblique", "reference", "measure", "front", "bed", "adjust",
                 "compare"],
        default="3d",
        help="表示モード。compare=ビフォー/アフター小顔チェック(ブラウザ), "
             "bed=ベッド基準化, adjust=矢状面の対話調整",
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
    p.add_argument("--p1", default=None,
                   help="[oblique] 切断ライン端点1 'X,Y'(mm)。未指定ならクリック")
    p.add_argument("--p2", default=None,
                   help="[oblique] 切断ライン端点2 'X,Y'(mm)")
    p.add_argument("--p3", default=None,
                   help="[reference] 基準面の3点目 'X,Y'(mm)。p1,p2,p3で平面定義")
    p.add_argument("--save-csv", default=None,
                   help="[reference] 基準座標に合わせた点群の保存先CSV")
    p.add_argument("--compare", default=None,
                   help="[measure] 比較する“後”のデータ(整列CSV等)。前後比較に使用")
    p.add_argument("--yaw", type=float, default=0.0,
                   help="[bed] 手動で水平回転(度)。矢状面の向き微調整")
    p.add_argument("--dx", type=float, default=0.0,
                   help="[bed] 手動で左右に移動(mm)。矢状面x'=0の位置調整")
    p.add_argument("--dy", type=float, default=0.0,
                   help="[bed] 手動で頭足方向に移動(mm)")
    p.add_argument("--no-center", action="store_true",
                   help="[bed] 自動の体中心合わせを無効化(手動のみで合わせる)")
    p.add_argument("--after", default=None,
                   help="[compare] アフターのデータ(.dat か CSV)")
    p.add_argument("--frame-before", type=int, default=None,
                   help="[compare] before が .dat のときのフレーム番号")
    p.add_argument("--frame-after", type=int, default=None,
                   help="[compare] after が .dat のときのフレーム番号")
    return p


def _parse_pt(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (float(a), float(b))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        path = resolve_path(args.path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.mode == "compare":
        if not args.after:
            print("compare には --after <アフターのファイル> が必要です。"
                  "同じ録画の別コマ同士なら --after same でOK。", file=sys.stderr)
            return 1
        after_path = path if args.after == "same" else resolve_path(args.after)
        from .compare_app import run_compare
        run_compare(path, after_path,
                    frame_before=args.frame_before or args.frame,
                    frame_after=args.frame_after)
        return 0

    if _is_dat(path):
        from . import toforge

        n = toforge.count_frames(path)
        if n == 0:
            print("このファイルは ToForge 形式として読めませんでした"
                  "(サイズが合いません)。", file=sys.stderr)
            return 1
        hdr = toforge.read_header(path, 0)
        print(f"[ToForge .dat] フレーム数: {n:,}  "
              f"(format {hdr['format_version']}, {toforge.SENSOR_WIDTH}x"
              f"{toforge.SENSOR_HEIGHT})")
        if args.list_frames:
            print(f"  --frame で 0〜{n-1} のフレームを選べます。"
                  f"例: --frame {n//2}")
            return 0

        if args.mode == "browse":
            from .browse import browse_frames
            browse_frames(path, start_frame=args.frame,
                          min_depth=args.min_depth, max_depth=args.max_depth)
            return 0

        if args.mode == "contact":
            from .browse import contact_sheet
            contact_sheet(path, n=args.n, start=args.start, end=args.end,
                          min_depth=args.min_depth, max_depth=args.max_depth,
                          save=args.save)
            return 0

        frame = args.frame if args.frame is not None else n // 2
        pc = toforge.read_frame(
            path, frame,
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
            path,
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
    elif args.mode == "oblique":
        from .oblique import oblique_slice
        ps = args.point_size or 6.0
        th = args.thickness if args.thickness is not None else 8.0
        oblique_slice(pc, p1=_parse_pt(args.p1), p2=_parse_pt(args.p2),
                      thickness=th, point_size=ps, save=args.save)
    elif args.mode == "bed":
        from .bed import align_to_bed
        align_to_bed(pc, center=not args.no_center, yaw=args.yaw,
                     dx=args.dx, dy=args.dy,
                     save_csv=args.save_csv, save=args.save)
    elif args.mode == "adjust":
        from .adjust_app import run_adjust
        run_adjust(pc, save_csv=args.save_csv)
    elif args.mode == "front":
        from .reference import save_front_map, save_front_html
        out = args.save or os.path.expanduser("~/Desktop/front_map.png")
        save_front_map(pc, out)
        html = os.path.splitext(out)[0] + ".html"
        try:
            save_front_html(pc, html)
        except Exception as e:  # plotly 無い等
            print(f"[front] HTML 版はスキップ: {e}", file=sys.stderr)
    elif args.mode == "reference":
        from .reference import define_reference
        pts = None
        if args.p1 and args.p2 and args.p3:
            pts = [_parse_pt(args.p1), _parse_pt(args.p2), _parse_pt(args.p3)]
        define_reference(pc, pts=pts, save_csv=args.save_csv, save=args.save)
    elif args.mode == "measure":
        from .measure import measure_and_plot
        pos = args.pos if args.pos is not None else 0.0
        th = args.thickness if args.thickness is not None else 5.0
        cmp_pc = None
        if args.compare:
            cmp_pc = load_points(resolve_path(args.compare))
        measure_and_plot(pc, axis=args.axis, position=pos, thickness=th,
                         label="before" if cmp_pc is not None else "data",
                         compare_pc=cmp_pc, compare_label="after",
                         save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
