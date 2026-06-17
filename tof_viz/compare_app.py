"""ビフォー/アフターを比べて「小顔になったか」を計測するブラウザアプリ.

2つの静止ToFデータ(.dat のフレーム、または整列CSV)を読み込み、
ベッド基準+矢状面中心に整列し、顔の幅・断面周囲・断面面積・体積を計測して
前後で比較。ブラウザ(Dash)でコマ(前/後)と高さをスライダーで選べる。
"""
from __future__ import annotations

import os
import webbrowser
from functools import lru_cache
from typing import Optional

import numpy as np

from .bed import compute_bed_aligned
from .loader import PointCloud, load_points
from .measure import measure_section


def _subject_mask(zp, frac=0.3):
    zmax = float(np.nanmax(zp)) if len(zp) else 1.0
    return zp > frac * zmax


def volume_above_bed(xp, yp, zp, *, res=5.0):
    """体(隆起部)のベッド上の体積 [cm^3] をグリッド積算で概算。"""
    m = _subject_mask(zp)
    if m.sum() < 50:
        return 0.0
    x, y, z = xp[m], yp[m], zp[m]
    ix = np.floor((x - x.min()) / res).astype(int)
    iy = np.floor((y - y.min()) / res).astype(int)
    ny = iy.max() + 1
    grid = np.zeros((ix.max() + 1) * ny)
    np.maximum.at(grid, ix * ny + iy, z)
    return float(grid.sum() * res * res / 1000.0)


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


def run_compare(before_path, after_path, *, frame_before=None,
                frame_after=None, port=8050):
    import dash
    from dash import Input, Output, dcc, html
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

    @lru_cache(maxsize=32)
    def load_align(path, frame, is_dat):
        if is_dat:
            from . import toforge
            pc = toforge.read_frame(path, frame, min_depth=1.0)
        else:
            pc = load_points(path)
        xp, yp, zp, inten, meta = compute_bed_aligned(pc, center=True)
        return xp, yp, zp

    fB0 = (frame_before if frame_before is not None
           else (nB // 2 if b_is_dat else 0))
    fA0 = (frame_after if frame_after is not None
           else (nA // 2 if a_is_dat else 0))

    def topfig(xp, yp, zp, title):
        m = _subject_mask(zp)
        fig = go.Figure(go.Scattergl(
            x=xp[m], y=yp[m], mode="markers",
            marker=dict(size=3, color=zp[m], colorscale="Turbo")))
        fig.add_vline(x=0, line_color="red")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title=title, height=340,
                          margin=dict(l=20, r=10, t=40, b=20))
        return fig

    def build(fB, fA, h):
        bx, by, bz = load_align(before_path, fB, b_is_dat)
        ax, ay, az = load_align(after_path, fA, a_is_dat)
        volB, volA = volume_above_bed(bx, by, bz), volume_above_bed(ax, ay, az)
        hi = min(float(bz.max()), float(az.max()))
        if h is None:
            h = 0.7 * hi
        h = min(h, hi)
        mB = face_metrics(bx, by, bz, height=h)
        mA = face_metrics(ax, ay, az, height=h)

        secf = go.Figure()
        if len(mB["curve"]):
            secf.add_trace(go.Scatter(x=mB["curve"][:, 0], y=mB["curve"][:, 1],
                           mode="lines+markers", name="before",
                           line_color="blue"))
        if len(mA["curve"]):
            secf.add_trace(go.Scatter(x=mA["curve"][:, 0], y=mA["curve"][:, 1],
                           mode="lines+markers", name="after",
                           line_color="red"))
        secf.update_yaxes(scaleanchor="x", scaleratio=1)
        secf.update_layout(title=f"高さ {h:.0f}mm の横断面(青=before 赤=after)",
                           height=400, xaxis_title="左右 x'[mm]",
                           yaxis_title="前後 y'[mm]")

        def trow(name, b, a, unit):
            d = a - b
            pct = (d / b * 100) if b else 0
            arrow = "⬇" if d < 0 else ("⬆" if d > 0 else "→")
            return html.Tr([html.Td(name), html.Td(f"{b:.1f}{unit}"),
                            html.Td(f"{a:.1f}{unit}"),
                            html.Td(f"{d:+.1f} ({pct:+.1f}%){arrow}")])
        table = html.Table([
            html.Thead(html.Tr([html.Th("項目"), html.Th("before"),
                                html.Th("after"), html.Th("変化")])),
            html.Tbody([
                trow("顔の幅(左右)", mB["width"], mA["width"], "mm"),
                trow("断面の周囲", mB["perimeter"], mA["perimeter"], "mm"),
                trow("断面の面積", mB["area"]/100, mA["area"]/100, "cm²"),
                trow("体積(ベッド上)", volB, volA, "cm³")])],
            style={"fontSize": "16px"})
        score = sum([mA["width"] < mB["width"], mA["area"] < mB["area"],
                     volA < volB])
        verdict = ("✅ 小顔になっています" if score >= 2 else
                   ("❌ 小顔になっていません" if score == 0 else
                    "△ 変化は小さい/まちまち"))
        top = html.Div([
            dcc.Graph(figure=topfig(bx, by, bz, f"before (frame {fB})"),
                      style={"display": "inline-block", "width": "49%"}),
            dcc.Graph(figure=topfig(ax, ay, az, f"after (frame {fA})"),
                      style={"display": "inline-block", "width": "49%"})])
        return top, secf, table, verdict, hi

    app = dash.Dash(__name__)
    top0, sec0, tbl0, ver0, hi0 = build(fB0, fA0, None)

    controls = []
    if b_is_dat:
        controls += [html.Label(f"before のコマ (0〜{nB-1})"),
                     dcc.Slider(0, nB - 1, max(1, nB // 200), value=fB0,
                                id="fb", tooltip={"placement": "bottom",
                                "always_visible": True})]
    if a_is_dat:
        controls += [html.Label(f"after のコマ (0〜{nA-1})"),
                     dcc.Slider(0, nA - 1, max(1, nA // 200), value=fA0,
                                id="fa", tooltip={"placement": "bottom",
                                "always_visible": True})]

    app.layout = html.Div([
        html.H2("ビフォー・アフター 小顔チェック"),
        html.Div(top0, id="top"),
        html.Div(controls),
        html.Label("計測する高さ z'(ベッドからの高さ mm)"),
        dcc.Slider(0, round(hi0), max(1, round(hi0 / 40)), value=round(0.7*hi0),
                   id="h", tooltip={"placement": "bottom",
                   "always_visible": True}),
        dcc.Graph(id="sec", figure=sec0),
        html.Div(tbl0, id="tbl"),
        html.H3(ver0, id="ver"),
        html.P("コマ/高さスライダーを動かすと再計算。終了はターミナルで Control+C。"),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    inputs = [Input("h", "value")]
    if b_is_dat:
        inputs.append(Input("fb", "value"))
    if a_is_dat:
        inputs.append(Input("fa", "value"))

    @app.callback(Output("top", "children"), Output("sec", "figure"),
                  Output("tbl", "children"), Output("ver", "children"),
                  *inputs)
    def _upd(*vals):
        h = vals[0]
        i = 1
        fB = vals[i] if b_is_dat else fB0
        i += 1 if b_is_dat else 0
        fA = vals[i] if a_is_dat else fA0
        top, sec, tbl, ver, _ = build(int(fB), int(fA), float(h))
        return top, sec, tbl, ver

    import socket
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    url = f"http://127.0.0.1:{port}"
    print(f"[compare] ブラウザで比較画面を開きます: {url}")
    print("  終了: ターミナルで Control+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(port=port, debug=False)
    except AttributeError:
        app.run_server(port=port, debug=False)
