"""ブラウザで動く 3D ビューア(どのSSD/フォルダからでもデータを選べる).

ToForge .dat の任意コマ、または CSV を、回転できる3D点群として表示する。
データ選択は比較アプリと同じ仕組み(/Volumes 配下の全SSD走査 + パス貼り付け)。
matplotlib のインタラクティブ表示が固まる Mac でも、ブラウザなら安定して動く。
"""
from __future__ import annotations

import os
import socket
import webbrowser
from functools import lru_cache

import numpy as np

from .compare_app import data_label, find_data_files, scan_folder
from .loader import load_points


def run_view3d(path, *, frame=None, port=8060):
    import dash
    from dash import Input, Output, State, dcc, html
    import plotly.graph_objects as go
    from . import toforge

    is_dat = path.lower().endswith(".dat")
    n = toforge.count_frames(path) if is_dat else 0
    # 実行中に差し替え可能な状態
    S = {"path": path, "is_dat": is_dat, "n": n}

    @lru_cache(maxsize=64)
    def load(path, frame, is_dat, max_depth, max_points):
        try:
            if is_dat:
                pc = toforge.read_frame(path, frame, min_depth=300.0,
                                        max_depth=max_depth,
                                        max_points=max_points)
            else:
                pc = load_points(path, max_points=max_points)
        except Exception:
            return None
        return pc

    def fig3d(pc, ptsize):
        fig = go.Figure()
        if pc is None or len(pc.xyz) == 0:
            fig.add_annotation(text="このコマには表示できる点がありません。<br>"
                               "コマや「遠くの背景を消す」を調整してください。",
                               showarrow=False, font=dict(size=16, color="red"),
                               xref="paper", yref="paper", x=0.5, y=0.5)
            fig.update_layout(height=700)
            return fig
        xyz = pc.xyz
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker=dict(size=ptsize, color=xyz[:, 2], colorscale="Turbo",
                        opacity=0.85)))
        fig.update_layout(
            height=720, margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode="data",
                       xaxis_title="X [mm]", yaxis_title="Y [mm]",
                       zaxis_title="距離 Z [mm]"))
        return fig

    fr0 = frame if frame is not None else (n // 2 if is_dat else 0)

    app = dash.Dash(__name__)

    def _opts(paths):
        return [{"label": data_label(p), "value": p} for p in paths]

    dat_opts = _opts(find_data_files(extra=[path]))

    app.layout = html.Div([
        html.H2("3D ビューア(ToFデータ)  [版 v4]"),
        html.Div([
            html.B("データの選択(どのSSDからでも)"),
            html.Label("計測データ(.dat / .csv)"),
            dcc.Dropdown(id="dd", options=dat_opts, value=path,
                         clearable=False),
            html.Button("▶ このデータを表示", id="loadbtn", n_clicks=0,
                        style={"fontSize": "16px", "marginTop": "8px",
                               "padding": "6px 14px"}),
            html.Div(id="loadmsg", style={"marginTop": "6px", "color": "#0a0"}),
            html.Hr(),
            html.Label("一覧に無いときは、SSD等のフォルダ/ファイルのパスを貼り付け"
                       "(例: /Volumes/SSD名/フォルダ)"),
            dcc.Input(id="pathbox", type="text", debounce=True,
                      placeholder="/Volumes/... を貼り付け",
                      style={"width": "70%"}),
            html.Button("このパスから探す", id="scanbtn", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Button("一覧を再スキャン", id="rescanbtn", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Div(id="scanmsg", style={"marginTop": "6px", "color": "#06c"}),
        ], style={"border": "2px solid #46a", "padding": "10px",
                  "margin": "8px 0", "background": "#eef4ff"}),
        html.Label("コマ(.datのとき)"),
        dcc.Slider(0, max(0, n - 1), max(1, (n // 200) or 1), value=fr0,
                   id="fr", tooltip={"placement": "bottom",
                                     "always_visible": True}),
        html.Label("遠くの背景を消す(最大距離 mm。小さくすると壁などが消える)"),
        dcc.Slider(700, 4000, 50, value=4000, id="maxd",
                   marks={700: "700", 1500: "1500", 2500: "2500", 4000: "全部"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("点の大きさ"),
        dcc.Slider(1, 5, 1, value=2, id="psz",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="g3d"),
        html.P("マウスでドラッグ=回転、ホイール=拡大縮小。終了はこのウインドウで"
               "Control+C。", style={"color": "#555"}),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    @app.callback(Output("g3d", "figure"),
                  Input("fr", "value"), Input("maxd", "value"),
                  Input("psz", "value"))
    def _update(fr, maxd, psz):
        fr = int(fr) if fr is not None else 0
        if S["is_dat"]:
            fr = min(max(0, fr), max(0, S["n"] - 1))
        md = None if (maxd is None or maxd >= 4000) else float(maxd)
        pc = load(S["path"], fr, S["is_dat"], md, 40000)
        return fig3d(pc, int(psz))

    @app.callback(
        Output("fr", "max"), Output("fr", "value"), Output("fr", "step"),
        Output("loadmsg", "children"),
        Input("loadbtn", "n_clicks"), State("dd", "value"),
        prevent_initial_call=True)
    def _reload(nclk, p):
        if not p:
            raise dash.exceptions.PreventUpdate
        S["path"] = p
        S["is_dat"] = p.lower().endswith(".dat")
        S["n"] = toforge.count_frames(p) if S["is_dat"] else 0
        nmax = max(0, S["n"] - 1)
        return (nmax, min(nmax, S["n"] // 2), max(1, (S["n"] // 200) or 1),
                f"表示中 → {os.path.basename(p)}")

    @app.callback(
        Output("dd", "options"), Output("scanmsg", "children"),
        Input("scanbtn", "n_clicks"), Input("rescanbtn", "n_clicks"),
        State("pathbox", "value"), prevent_initial_call=True)
    def _scan(n_scan, n_rescan, pathstr):
        tid = (dash.callback_context.triggered[0]["prop_id"]
               if dash.callback_context.triggered else "")
        deep = tid.startswith("rescanbtn")
        paths = find_data_files(extra=[S["path"]], deep=deep)
        msg = f"自動一覧: {len(paths)} 件"
        if tid.startswith("scanbtn"):
            extra = scan_folder(pathstr or "")
            if not extra:
                msg = (f"『{pathstr}』からデータ(.dat/.csv)が見つかりません。"
                       "パスが正しいか確認してください。")
            else:
                msg = f"『{pathstr}』から {len(extra)} 件見つかりました。"
            paths = list(dict.fromkeys(extra + paths))
        return _opts(paths), msg

    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    url = f"http://127.0.0.1:{port}"
    lan_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = None
    print(f"[view3d] このMacで開く: {url}")
    if lan_ip and not lan_ip.startswith("127."):
        print(f"[view3d] iPhone/iPadで開く(同じWi-Fi): http://{lan_ip}:{port}")
    print("  ドラッグで回転、ホイールで拡大。終了: Control+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except AttributeError:
        app.run_server(host="0.0.0.0", port=port, debug=False)
