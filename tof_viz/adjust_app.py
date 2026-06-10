"""ブラウザ(Dash)で矢状面の向き・位置を対話的に調整するアプリ.

matplotlib の窓が固まる環境向け。Chrome 等のブラウザでスライダーを動かすと、
その場で回転・移動して赤い中線(矢状面 x'=0)が体の中心に合う様子が見える。
「保存」ボタンで整列済み点群を CSV 出力する。
"""
from __future__ import annotations

import os
import webbrowser
from typing import Optional

import numpy as np

from .bed import apply_manual, compute_bed_aligned
from .loader import PointCloud


def run_adjust(pc: PointCloud, *, save_csv: Optional[str] = None,
               port: int = 8050, max_points: int = 25000):
    import dash
    from dash import Input, Output, State, dcc, html
    import plotly.graph_objects as go

    if save_csv is None:
        save_csv = os.path.expanduser("~/Desktop/aligned_bed.csv")

    # ベッド+自動中心合わせ(基準)を一度だけ計算
    xp0, yp0, zp0, inten0, meta = compute_bed_aligned(pc, center=True)

    # 表示用に間引き
    rng = np.random.default_rng(0)
    if len(xp0) > max_points:
        idx = rng.choice(len(xp0), max_points, replace=False)
    else:
        idx = np.arange(len(xp0))
    vx, vy, vz = xp0[idx], yp0[idx], zp0[idx]
    zlo, zhi = np.percentile(zp0, [1, 99])

    def make_fig(yaw, dx, dy):
        ax, ay = apply_manual(vx.copy(), vy.copy(), yaw, dx, dy)
        fig = go.Figure(go.Scattergl(
            x=ax, y=ay, mode="markers",
            marker=dict(size=3, color=np.clip(vz, zlo, zhi),
                        colorscale="Turbo", colorbar=dict(title="height mm")),
            hovertemplate="x'=%{x:.0f}<br>y'=%{y:.0f}<extra></extra>"))
        # 矢状中線(x'=0)
        fig.add_vline(x=0, line_color="red", line_width=2)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            title="TOP view — red line = sagittal midline (x'=0). "
                  "Adjust sliders so it runs through the body center.",
            xaxis_title="x' left-right [mm]", yaxis_title="y' head-foot [mm]",
            width=820, height=760, margin=dict(l=40, r=40, t=60, b=40))
        return fig

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H3("矢状面の調整(赤線を体の中心に合わせる)"),
        dcc.Graph(id="g", figure=make_fig(0, 0, 0)),
        html.Div([
            html.Label("回転 yaw [度]"),
            dcc.Slider(-45, 45, 0.5, value=0, id="yaw",
                       marks={-45: "-45", 0: "0", 45: "45"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("左右 dx [mm]"),
            dcc.Slider(-250, 250, 1, value=0, id="dx",
                       marks={-250: "-250", 0: "0", 250: "250"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("頭足 dy [mm]"),
            dcc.Slider(-250, 250, 1, value=0, id="dy",
                       marks={-250: "-250", 0: "0", 250: "250"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"width": "820px"}),
        html.Button("この向きで保存(CSV)", id="save", n_clicks=0,
                    style={"fontSize": "18px", "padding": "10px",
                           "marginTop": "10px"}),
        html.Div(id="status", style={"marginTop": "10px", "fontSize": "16px"}),
        html.P("調整できたら「保存」を押す → ターミナルで Control+C を押して終了。"),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    @app.callback(Output("g", "figure"),
                  Input("yaw", "value"), Input("dx", "value"),
                  Input("dy", "value"))
    def _update(yaw, dx, dy):
        return make_fig(yaw or 0, dx or 0, dy or 0)

    @app.callback(Output("status", "children"), Input("save", "n_clicks"),
                  State("yaw", "value"), State("dx", "value"),
                  State("dy", "value"), prevent_initial_call=True)
    def _save(n, yaw, dx, dy):
        ax, ay = apply_manual(xp0.copy(), yp0.copy(), yaw or 0, dx or 0, dy or 0)
        out = np.column_stack([ax, ay, zp0, inten0])
        np.savetxt(save_csv, out, delimiter=",", header="x,y,z,intensity",
                   comments="", fmt="%.3f")
        msg = (f"保存しました: {save_csv}  (yaw={yaw:+.1f}°, dx={dx:+.0f}, "
               f"dy={dy:+.0f} mm)")
        print("[adjust] " + msg)
        return msg

    url = f"http://127.0.0.1:{port}"
    print(f"[adjust] ブラウザで調整画面を開きます: {url}")
    print("[adjust] スライダーで赤線を体の中心に合わせ→「保存」→ "
          "ターミナルで Control+C で終了。")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(port=port, debug=False)
    except AttributeError:
        app.run_server(port=port, debug=False)
