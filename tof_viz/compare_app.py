"""ビフォー/アフターを比べて「小顔になったか」を計測するブラウザアプリ.

2つの静止ToFデータ(.dat のフレーム、または整列CSV)を読み込み、
ベッド基準+矢状面中心に整列し、顔の幅・断面周囲・断面面積・体積を計測して
前後で比較。ブラウザ(Dash)で高さスライダーを動かしながら見られる。
"""
from __future__ import annotations

import os
import webbrowser
from typing import Optional

import numpy as np

from .bed import compute_bed_aligned
from .loader import PointCloud, load_points
from .measure import measure_section


def _subject_mask(zp, frac=0.3):
    zmax = float(np.nanmax(zp))
    return zp > frac * zmax


def volume_above_bed(xp, yp, zp, *, res=5.0):
    """体(隆起部)のベッド上の体積 [cm^3] をグリッド積算で概算。"""
    m = _subject_mask(zp)
    if m.sum() < 50:
        return 0.0
    x, y, z = xp[m], yp[m], zp[m]
    ix = np.floor((x - x.min()) / res).astype(int)
    iy = np.floor((y - y.min()) / res).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    grid = np.zeros((nx, ny))
    flat = ix * ny + iy
    np.maximum.at(grid.ravel(), flat, z)        # 各マスの最大高さ
    vol_mm3 = grid.sum() * res * res            # Σ 高さ×マス面積
    return float(vol_mm3 / 1000.0)              # cm^3


def face_metrics(xp, yp, zp, *, height):
    """指定高さ(ベッドからの z')の横断面の 幅/周囲/面積 を返す[mm, mm, mm^2]。"""
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


def _load_any(path, frame=None, min_depth=1.0, max_depth=None):
    """.dat ならフレームを、CSV なら点群を読み込み、ベッド整列して返す。"""
    if path.lower().endswith(".dat"):
        from . import toforge
        n = toforge.count_frames(path)
        fr = n // 2 if frame is None else frame
        pc = toforge.read_frame(path, fr, min_depth=min_depth,
                                max_depth=max_depth)
    else:
        pc = load_points(path)
    xp, yp, zp, inten, meta = compute_bed_aligned(pc, center=True)
    return xp, yp, zp


def run_compare(before_path, after_path, *, frame_before=None,
                frame_after=None, port=8050):
    import dash
    from dash import Input, Output, dcc, html
    import plotly.graph_objects as go

    print("[compare] before を読み込み・整列中...")
    bx, by, bz = _load_any(before_path, frame_before)
    print("[compare] after を読み込み・整列中...")
    ax, ay, az = _load_any(after_path, frame_after)

    volB = volume_above_bed(bx, by, bz)
    volA = volume_above_bed(ax, ay, az)

    # 顔の高さ範囲(共通)を決める
    zmaxB, zmaxA = float(bz.max()), float(az.max())
    hi = min(zmaxB, zmaxA)
    lo = 0.4 * hi
    h0 = 0.7 * hi

    def topfig(xp, yp, zp, title):
        m = _subject_mask(zp)
        fig = go.Figure(go.Scattergl(
            x=xp[m], y=yp[m], mode="markers",
            marker=dict(size=3, color=zp[m], colorscale="Turbo",
                        colorbar=dict(title="高さmm"))))
        fig.add_vline(x=0, line_color="red")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title=title, height=380,
                          margin=dict(l=30, r=10, t=40, b=30))
        return fig

    def compare_section(h):
        mB = face_metrics(bx, by, bz, height=h)
        mA = face_metrics(ax, ay, az, height=h)
        fig = go.Figure()
        if len(mB["curve"]):
            fig.add_trace(go.Scatter(x=mB["curve"][:, 0], y=mB["curve"][:, 1],
                          mode="lines+markers", name="before", line_color="blue"))
        if len(mA["curve"]):
            fig.add_trace(go.Scatter(x=mA["curve"][:, 0], y=mA["curve"][:, 1],
                          mode="lines+markers", name="after", line_color="red"))
        fig.add_vline(x=0, line_color="gray", line_dash="dot")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title=f"高さ {h:.0f}mm の横断面(青=before, 赤=after)",
                          height=420, xaxis_title="左右 x' [mm]",
                          yaxis_title="前後 y' [mm]")
        return fig, mB, mA

    app = dash.Dash(__name__)

    def metrics_table(mB, mA):
        rows = []
        def row(name, b, a, unit, smaller_is_better=True):
            d = a - b
            pct = (d / b * 100) if b else 0
            arrow = "⬇小さく" if d < 0 else ("⬆大きく" if d > 0 else "→")
            return html.Tr([html.Td(name),
                            html.Td(f"{b:.1f} {unit}"),
                            html.Td(f"{a:.1f} {unit}"),
                            html.Td(f"{d:+.1f} ({pct:+.1f}%) {arrow}")])
        rows.append(row("顔の幅(左右)", mB["width"], mA["width"], "mm"))
        rows.append(row("断面の周囲", mB["perimeter"], mA["perimeter"], "mm"))
        rows.append(row("断面の面積", mB["area"]/100, mA["area"]/100, "cm²"))
        rows.append(row("体積(ベッド上)", volB, volA, "cm³"))
        return html.Table([
            html.Thead(html.Tr([html.Th("項目"), html.Th("before"),
                                html.Th("after"), html.Th("変化")])),
            html.Tbody(rows)],
            style={"borderCollapse": "collapse", "width": "700px"},
            className="mtable")

    def verdict(mB, mA):
        score = sum([mA["width"] < mB["width"], mA["area"] < mB["area"],
                     volA < volB])
        if score >= 2:
            return "✅ 小顔になっています(幅・面積・体積の多くが減少)"
        if score == 0:
            return "❌ 小顔にはなっていません(増加傾向)"
        return "△ 変化は小さい/まちまちです"

    fig0, mB0, mA0 = compare_section(h0)
    app.layout = html.Div([
        html.H2("ビフォー・アフター 小顔チェック"),
        html.Div([
            dcc.Graph(figure=topfig(bx, by, bz, "before(上から)"),
                      style={"display": "inline-block", "width": "49%"}),
            dcc.Graph(figure=topfig(ax, ay, az, "after(上から)"),
                      style={"display": "inline-block", "width": "49%"}),
        ]),
        html.Label("計測する高さ z'(ベッドからの高さ mm)"),
        dcc.Slider(round(lo), round(hi), max(1, round((hi-lo)/40)),
                   value=round(h0), id="h",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="sec", figure=fig0),
        html.Div(id="tbl", children=metrics_table(mB0, mA0)),
        html.H3(id="verdict", children=verdict(mB0, mA0)),
        html.P("高さスライダーを動かすと、その高さの断面で比較します。"
               "終わるには、ターミナルで Control+C。"),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    @app.callback(Output("sec", "figure"), Output("tbl", "children"),
                  Output("verdict", "children"), Input("h", "value"))
    def _upd(h):
        fig, mB, mA = compare_section(float(h))
        return fig, metrics_table(mB, mA), verdict(mB, mA)

    # 空きポート
    import socket
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    url = f"http://127.0.0.1:{port}"
    print(f"[compare] ブラウザで比較画面を開きます: {url}")
    print(f"  before vol={volB:.0f}cm³, after vol={volA:.0f}cm³")
    print("  終了: ターミナルで Control+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(port=port, debug=False)
    except AttributeError:
        app.run_server(port=port, debug=False)
