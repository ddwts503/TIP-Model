"""ビフォー/アフターを比べて「小顔になったか」を計測するブラウザアプリ.

2つの静止ToFデータ(.dat のフレーム/整列CSV)をベッド基準に整列し、
ユーザが画像上で顔の中心をクリック→その周りを顔として切り出し、
顔の幅・断面周囲・断面面積・体積を前後で比較する。位置合わせ(ICP)も可。
"""
from __future__ import annotations

import webbrowser
from functools import lru_cache
from typing import Optional

import numpy as np

from .bed import compute_bed_aligned
from .loader import PointCloud, load_points
from .measure import measure_section


def _subject_mask(zp, frac=0.3):
    """ベッドより十分高い点(=体)。外れ値に強いよう 99.5%tile を上限に。"""
    if len(zp) == 0:
        return np.zeros(0, bool)
    zmax = float(np.percentile(zp, 99.5))
    return zp > frac * zmax


def register_icp(src_xyz, tgt_xyz, max_dist=25.0, iters=40):
    """剛体ICP(numpy/scipyのみ)。after を before に重ねる(拡縮なし)。"""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return src_xyz
    if len(src_xyz) < 30 or len(tgt_xyz) < 30:
        return src_xyz
    src = src_xyz.astype(float).copy()
    tgt = tgt_xyz.astype(float)
    tree = cKDTree(tgt)
    for _ in range(iters):
        d, idx = tree.query(src, workers=-1)
        keep = d < max_dist
        if int(keep.sum()) < 20:
            break
        P, Q = src[keep], tgt[idx[keep]]
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        new = src @ R.T + (qc - R @ pc)
        if np.abs(new - src).max() < 1e-3:
            src = new
            break
        src = new
    return src


def _auto_nose(xp, yp, zp):
    """鼻=一番手前(高さ最大)の位置を自動検出(外れ値に強い中央値)。"""
    if len(zp) == 0:
        return (0.0, 0.0)
    m = zp >= np.percentile(zp, 98)
    return (float(np.median(xp[m])), float(np.median(yp[m])))


def _crop(xp, yp, zp, center, radius):
    """center=(cx,cy) の周り半径 radius[mm] だけ残す。center None なら鼻を自動検出。"""
    if center is None:
        center = _auto_nose(xp, yp, zp)
    cx, cy = center
    keep = (xp - cx) ** 2 + (yp - cy) ** 2 <= radius ** 2
    return xp[keep], yp[keep], zp[keep], center


def face_dims(xp, yp, zp):
    """切り出した顔の主要寸法を返す(外れ値に強い 2〜98 パーセンタイル幅)。

    width=左右(顔幅), length=頭足(顔の長さ), depth=突出(奥行き=高さ方向)。
    """
    def ext(v):
        if len(v) < 5:
            return 0.0
        return float(np.percentile(v, 98) - np.percentile(v, 2))
    return ext(xp), ext(yp), ext(zp)


def face_metrics(xp, yp, zp, *, height):
    pc = PointCloud(xyz=np.column_stack([xp, yp, zp]))
    try:
        length, area, curve, _ = measure_section(
            pc, axis="z", position=height, thickness=8.0)
        width = float(curve[:, 0].max() - curve[:, 0].min())
        return {"width": width, "perimeter": length, "area": area,
                "curve": curve}
    except Exception:
        return {"width": 0.0, "perimeter": 0.0, "area": 0.0,
                "curve": np.empty((0, 2))}


def volume_face(xp, yp, zp, base, *, res=4.0):
    """切り出した顔の、base[mm]より上の体積[cm^3](顔だけの盛り上がり)。"""
    if len(zp) < 30:
        return 0.0
    h = zp - base
    h = np.clip(h, 0, None)
    ix = np.floor((xp - xp.min()) / res).astype(int)
    iy = np.floor((yp - yp.min()) / res).astype(int)
    ny = iy.max() + 1
    grid = np.zeros((ix.max() + 1) * ny)
    np.maximum.at(grid, ix * ny + iy, h)
    return float(grid.sum() * res * res / 1000.0)


def run_compare(before_path, after_path, *, frame_before=None,
                frame_after=None, port=8050):
    import dash
    from dash import Input, Output, State, dcc, html
    import plotly.graph_objects as go

    b_is_dat = before_path.lower().endswith(".dat")
    a_is_dat = after_path.lower().endswith(".dat")
    nB = nA = 0
    if b_is_dat or a_is_dat:
        from . import toforge
        if b_is_dat:
            nB = toforge.count_frames(before_path)
        if a_is_dat:
            nA = toforge.count_frames(after_path)

    @lru_cache(maxsize=64)
    def load_align(path, frame, is_dat):
        """カメラそのままの向き(顔は正立)。人物だけ残し、鼻=手前を高さに。"""
        if is_dat:
            from . import toforge
            pc = toforge.read_frame(path, frame, min_depth=300.0)
        else:
            pc = load_points(path)
        x, y, z = np.asarray(pc.x), np.asarray(pc.y), np.asarray(pc.z)
        if len(z) == 0:
            return x, y, z
        znear = float(np.percentile(z, 1))       # 最も手前(鼻側)
        keep = (z >= znear) & (z <= znear + 350)  # 人物だけ(背景/ベッド除去)
        x, y, z = x[keep], y[keep], z[keep]
        height = float(z.max()) - z               # 手前(鼻)ほど大きい高さ
        return x, y, height

    fB0 = frame_before if frame_before is not None else (nB // 2 if b_is_dat else 0)
    fA0 = frame_after if frame_after is not None else (nA // 2 if a_is_dat else 0)

    def topfig(xp, yp, zp, title, center, radius):
        m = _subject_mask(zp)
        fig = go.Figure(go.Scattergl(
            x=xp[m], y=yp[m], mode="markers",
            marker=dict(size=3, color=zp[m], colorscale="Turbo")))
        cx, cy = center
        th = np.linspace(0, 2 * np.pi, 80)
        fig.add_trace(go.Scatter(x=cx + radius * np.cos(th),
                                 y=cy + radius * np.sin(th), mode="lines",
                                 line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                                 marker=dict(color="black", size=10,
                                             symbol="x")))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title=title, height=380, showlegend=False,
                          margin=dict(l=20, r=10, t=40, b=20))
        return fig

    def compute(fB, fA, h, radius, register, cB, cA):
        bx, by, bz = load_align(before_path, fB, b_is_dat)
        ax, ay, az = load_align(after_path, fA, a_is_dat)
        bx, by, bz, cB = _crop(bx, by, bz, cB, radius)
        ax, ay, az, cA = _crop(ax, ay, az, cA, radius)
        if register and len(bz) > 20 and len(az) > 20:
            reg = register_icp(np.column_stack([ax, ay, az]),
                               np.column_stack([bx, by, bz]))
            ax, ay, az = reg[:, 0], reg[:, 1], reg[:, 2]
        baseB = float(np.percentile(bz, 10)) if len(bz) else 0.0
        baseA = float(np.percentile(az, 10)) if len(az) else 0.0
        volB = volume_face(bx, by, bz, baseB)
        volA = volume_face(ax, ay, az, baseA)
        hi = min(float(bz.max()), float(az.max())) if len(bz) and len(az) else 1.0
        if h is None:
            h = 0.6 * float(bz.max()) if len(bz) else 1.0
        h = min(h, hi)
        mB = face_metrics(bx, by, bz, height=h)
        mA = face_metrics(ax, ay, az, height=h)
        return (bx, by, bz, ax, ay, az, mB, mA, volB, volA, hi, h)

    def make_outputs(state):
        (bx, by, bz, ax, ay, az, mB, mA, volB, volA, hi, h) = state
        secf = go.Figure()
        if len(mB["curve"]):
            secf.add_trace(go.Scatter(x=mB["curve"][:, 0], y=mB["curve"][:, 1],
                           mode="lines+markers", name="before", line_color="blue"))
        if len(mA["curve"]):
            secf.add_trace(go.Scatter(x=mA["curve"][:, 0], y=mA["curve"][:, 1],
                           mode="lines+markers", name="after", line_color="red"))
        secf.update_yaxes(scaleanchor="x", scaleratio=1)
        secf.update_layout(title=f"高さ {h:.0f}mm の断面(青=before 赤=after)",
                           height=380, xaxis_title="x'[mm]", yaxis_title="y'[mm]")
        wB, lB, dB = face_dims(bx, by, bz)
        wA, lA, dA = face_dims(ax, ay, az)

        def trow(name, b, a, unit, key=False):
            d = a - b
            pct = (d / b * 100) if b else 0
            arrow = "⬇小さく" if d < 0 else ("⬆大きく" if d > 0 else "→")
            style = {"fontWeight": "bold"} if key else {}
            return html.Tr([html.Td(name, style=style),
                            html.Td(f"{b:.1f}{unit}"), html.Td(f"{a:.1f}{unit}"),
                            html.Td(f"{d:+.1f} ({pct:+.1f}%){arrow}", style=style)])
        table = html.Table([
            html.Thead(html.Tr([html.Th("項目"), html.Th("前"), html.Th("後"),
                                html.Th("変化")])),
            html.Tbody([
                trow("◎ 顔幅(左右)", wB, wA, "mm", key=True),
                trow("◎ 奥行き(突出・高さ)", dB, dA, "mm", key=True),
                trow("顔の長さ(頭足)", lB, lA, "mm"),
                trow("顔の体積", volB, volA, "cm³"),
                trow("(参考)断面の面積", mB["area"]/100, mA["area"]/100, "cm²")])],
            style={"fontSize": "16px"})
        score = sum([wA < wB, dA < dB, volA < volB])
        verdict = ("✅ 小顔になっています(幅・奥行きが減少)" if score >= 2 else
                   ("❌ 小顔になっていません" if score == 0 else
                    "△ まちまち(幅は減・奥行きは増 など)"))
        return secf, table, verdict

    # --- 顔の初期中心(被写体の重心)とスライダー範囲を決める ---
    bxf, byf, bzf = load_align(before_path, fB0, b_is_dat)
    axf, ayf, azf = load_align(after_path, fA0, a_is_dat)

    def _center0(x, y):
        """初期円の位置=人物のだいたい中央(あとでドラッグで合わせる)。"""
        if len(x) == 0:
            return 0.0, 0.0
        return float(np.median(x)), float(np.median(y))
    dbx, dby = _center0(bxf, byf)
    dax, day = _center0(axf, ayf)
    # スライダー範囲は人物の点から(±1500に制限)
    allx = np.concatenate([bxf, axf]) if len(bxf) and len(axf) else np.array([0.0])
    ally = np.concatenate([byf, ayf]) if len(byf) and len(ayf) else np.array([0.0])
    xlo = max(-1500.0, float(np.percentile(allx, 1)) - 100)
    xhi = min(1500.0, float(np.percentile(allx, 99)) + 100)
    ylo = max(-1500.0, float(np.percentile(ally, 1)) - 100)
    yhi = min(1500.0, float(np.percentile(ally, 99)) + 100)

    app = dash.Dash(__name__)

    def cslider(id_, lo, hi, val):
        return dcc.Slider(round(lo), round(hi), 2, value=round(val), id=id_,
                          tooltip={"placement": "bottom", "always_visible": True})

    frame_ctrls = []
    if b_is_dat:
        frame_ctrls += [html.Label(f"前のコマ (0〜{nB-1})"),
                        dcc.Slider(0, nB - 1, max(1, nB // 200), value=fB0,
                                   id="fb", tooltip={"placement": "bottom",
                                   "always_visible": True})]
    if a_is_dat:
        frame_ctrls += [html.Label(f"後のコマ (0〜{nA-1})"),
                        dcc.Slider(0, nA - 1, max(1, nA // 200), value=fA0,
                                   id="fa", tooltip={"placement": "bottom",
                                   "always_visible": True})]

    def cslider(id_, lo, hi, val):
        return dcc.Slider(round(lo), round(hi), 5, value=round(val), id=id_,
                          tooltip={"placement": "bottom", "always_visible": True})

    app.layout = html.Div([
        html.H2("ビフォー・アフター 小顔チェック  [版 v4 スライダー]"),
        html.P("黒い円(×が中心)を顔に合わせます。下の『円 左右/上下』スライダーを"
               "動かして、左=before・右=after それぞれ×を顔の鼻に合わせてください。"),
        html.Div([
            dcc.Graph(id="gb", style={"display": "inline-block", "width": "49%"}),
            dcc.Graph(id="ga", style={"display": "inline-block", "width": "49%"})]),
        html.Div(frame_ctrls),
        html.B("before: 円 左右 / 上下"),
        cslider("bxc", xlo, xhi, dbx), cslider("byc", ylo, yhi, dby),
        html.B("after: 円 左右 / 上下"),
        cslider("axc", xlo, xhi, dax), cslider("ayc", ylo, yhi, day),
        dcc.Checklist(id="reg", options=[{"label": " 2つの顔を自動で重ねて比較",
                      "value": "on"}], value=["on"], style={"fontSize": "16px"}),
        html.Label("顔の範囲(半径 mm)= 円の大きさ"),
        dcc.Slider(40, 200, 5, value=100, id="r",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("計測する高さ z'(mm)"),
        dcc.Slider(0, 200, 2, value=60, id="h",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="sec"),
        html.Div(id="tbl"),
        html.H3(id="ver"),
        html.P("円を顔に合わせ→範囲/高さを調整→表と判定を確認。終了はControl+C。"),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    outs = [Output("gb", "figure"), Output("ga", "figure"),
            Output("sec", "figure"), Output("tbl", "children"),
            Output("ver", "children")]
    ins = [Input("r", "value"), Input("h", "value"), Input("reg", "value"),
           Input("bxc", "value"), Input("byc", "value"),
           Input("axc", "value"), Input("ayc", "value")]
    if b_is_dat:
        ins.append(Input("fb", "value"))
    if a_is_dat:
        ins.append(Input("fa", "value"))

    @app.callback(*outs, *ins)
    def _update(r, h, reg, bxc, byc, axc, ayc, *frames):
        i = 0
        fB = frames[i] if b_is_dat else fB0
        i += 1 if b_is_dat else 0
        fA = frames[i] if a_is_dat else fA0
        cB = (float(bxc), float(byc))
        cA = (float(axc), float(ayc))
        st = compute(int(fB), int(fA), float(h), float(r), bool(reg), cB, cA)
        bxx, byy, bzz, axx, ayy, azz = (load_align(before_path, int(fB), b_is_dat)
                                        + load_align(after_path, int(fA), a_is_dat))
        gb = topfig(bxx, byy, bzz, f"before (frame {fB})", cB, float(r))
        ga = topfig(axx, ayy, azz, f"after (frame {fA})", cA, float(r))
        secf, table, verdict = make_outputs(st)
        return gb, ga, secf, table, verdict

    import socket
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    url = f"http://127.0.0.1:{port}"
    print(f"[compare] ブラウザで比較画面を開きます: {url}")
    print("  顔をクリックして範囲を合わせてください。終了: Control+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(port=port, debug=False)
    except AttributeError:
        app.run_server(port=port, debug=False)
