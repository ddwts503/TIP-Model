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

from .compare_app import (
    data_label, find_data_files, list_volumes, open_browser, scan_folder,
    _write_url_file)
from .loader import load_points

_open_browser = open_browser


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

    def _clip(pc):
        """外れ値(遠くにポツンとある点)を除いた xyz, intensity を返す。"""
        xyz = pc.xyz
        inten = pc.intensity if pc.intensity is not None else xyz[:, 2]
        if len(xyz) > 50:
            m = np.ones(len(xyz), bool)
            for j in range(3):
                lo, hi = np.percentile(xyz[:, j], [0.5, 99.5])
                m &= (xyz[:, j] >= lo) & (xyz[:, j] <= hi)
            if int(m.sum()) > 50:
                xyz, inten = xyz[m], inten[m]
        return xyz, inten

    def _box(rangeon, rx, ry, rz):
        if not rangeon or "on" not in rangeon:
            return None
        return {"x": rx, "y": ry, "z": rz}

    def _apply_box(xyz, inten, box):
        if not box:
            return xyz, inten
        m = ((xyz[:, 0] >= box["x"][0]) & (xyz[:, 0] <= box["x"][1]) &
             (xyz[:, 1] >= box["y"][0]) & (xyz[:, 1] <= box["y"][1]) &
             (xyz[:, 2] >= box["z"][0]) & (xyz[:, 2] <= box["z"][1]))
        if int(m.sum()) == 0:
            return xyz, inten              # 空になるなら無視(全体を表示)
        return xyz[m], inten[m]

    def _make_gif(xyz, inten, colorby):
        """点群を1回転させた回転GIF(base64)を作る。"""
        import base64
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        if len(xyz) > 9000:
            idx = np.random.default_rng(0).choice(len(xyz), 9000, replace=False)
            xyz, inten = xyz[idx], inten[idx]
        c = inten if colorby == "ir" else xyz[:, 2]
        cmap = "gray" if colorby == "ir" else "turbo"
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=cmap, s=2)
        try:
            ax.set_box_aspect((np.ptp(xyz[:, 0]) + 1, np.ptp(xyz[:, 1]) + 1,
                               np.ptp(xyz[:, 2]) + 1))
        except Exception:
            pass
        ax.set_axis_off()

        def upd(a):
            ax.view_init(elev=20, azim=a)
        anim = FuncAnimation(fig, upd, frames=range(0, 360, 20))
        tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        tmp.close()
        try:
            anim.save(tmp.name, writer=PillowWriter(fps=8))
            with open(tmp.name, "rb") as f:
                data = f.read()
        finally:
            plt.close(fig)
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        return base64.b64encode(data).decode()

    def fig3d(pc, ptsize, colorby="z", box=None):
        fig = go.Figure()
        if pc is None or len(pc.xyz) == 0:
            fig.add_annotation(text="このコマには表示できる点がありません。<br>"
                               "コマや「遠くの背景を消す」を調整してください。",
                               showarrow=False, font=dict(size=16, color="red"),
                               xref="paper", yref="paper", x=0.5, y=0.5)
            fig.update_layout(height=480)
            return fig
        xyz, inten = _clip(pc)
        xyz, inten = _apply_box(xyz, inten, box)
        if colorby == "ir":          # 明るさ(IR)=実写ふう
            col, cs = inten, "Gray"
        else:                         # 距離(高さ)=色分け
            col, cs = xyz[:, 2], "Turbo"
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker=dict(size=ptsize, color=col, colorscale=cs,
                        opacity=0.85)))
        # データの範囲に合わせて自動ズーム(外れ値で小さくならないように)
        rng = {}
        for j, key in ((0, "xaxis"), (1, "yaxis"), (2, "zaxis")):
            lo, hi = float(xyz[:, j].min()), float(xyz[:, j].max())
            pad = (hi - lo) * 0.05 + 1
            rng[key] = dict(range=[lo - pad, hi + pad])
        fig.update_layout(
            height=480, margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode="data",
                       xaxis=dict(title="X [mm]", **rng["xaxis"]),
                       yaxis=dict(title="Y [mm]", **rng["yaxis"]),
                       zaxis=dict(title="距離 Z [mm]", **rng["zaxis"])))
        return fig

    fr0 = frame if frame is not None else (n // 2 if is_dat else 0)

    app = dash.Dash(__name__)

    def _opts(paths):
        return [{"label": data_label(p), "value": p} for p in paths]

    dat_opts = _opts(find_data_files(extra=[path]))

    # 範囲選択スライダーの上下限は、最初のコマの人の広がりから決める
    _seed = load(path, fr0, is_dat, 1500.0, 40000)
    if _seed is not None and len(_seed.xyz):
        _sx, _ = _clip(_seed)
        ext = {ax: (round(float(_sx[:, j].min())), round(float(_sx[:, j].max())))
               for j, ax in ((0, "x"), (1, "y"), (2, "z"))}
    else:
        ext = {"x": (-1000, 1000), "y": (-1000, 1000), "z": (0, 2000)}

    def _rslider(id_, lo, hi):
        return dcc.RangeSlider(lo, hi, max(1, (hi - lo) // 100), value=[lo, hi],
                               id=id_, tooltip={"placement": "bottom",
                                                "always_visible": True})

    app.layout = html.Div([
        html.H2("3D ビューア(ToFデータ)  [版 v14]"),
        html.Div([
            html.B("データの選択(どのSSDからでも)"),
            html.Label("① SSD(ドライブ)を選ぶ"),
            dcc.Dropdown(id="vol",
                         options=[{"label": "🟦 " + os.path.basename(p),
                                   "value": p} for p in list_volumes()],
                         placeholder="つないでいるSSDを選ぶと、中のデータが下に出ます",
                         clearable=False),
            html.Label("② 計測データ(.dat / .csv)を選ぶ"),
            dcc.Dropdown(id="dd", options=dat_opts, value=path,
                         clearable=False),
            html.Button("▶ このデータを表示", id="loadbtn", n_clicks=0,
                        style={"fontSize": "16px", "marginTop": "8px",
                               "padding": "6px 14px"}),
            html.Div(id="loadmsg",
                     children="現在のデータ: " + data_label(path),
                     style={"marginTop": "8px", "fontSize": "17px",
                            "fontWeight": "bold", "color": "#063"}),
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
        dcc.Slider(700, 4000, 50, value=1500, id="maxd",
                   marks={700: "700", 1500: "1500", 2500: "2500", 4000: "全部"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("点の大きさ"),
        dcc.Slider(1, 5, 1, value=2, id="psz",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div([
            html.Span("色: ", style={"marginRight": "6px"}),
            dcc.RadioItems(id="colorby", value="z", inline=True,
                           options=[{"label": " 距離(色分け)", "value": "z"},
                                    {"label": " 明るさ(実写ふう)",
                                     "value": "ir"}],
                           style={"display": "inline-block",
                                  "marginRight": "16px"}),
            html.Button("↺ 視点リセット", id="resetbtn", n_clicks=0,
                        style={"marginRight": "8px"}),
            html.Button("⬇ CSV書き出し", id="csvbtn", n_clicks=0,
                        style={"marginRight": "8px"}),
            html.Button("🎥 録画(回転GIF)", id="recbtn", n_clicks=0),
            dcc.Download(id="dl"),
            html.Span(id="recmsg", style={"marginLeft": "10px",
                                          "color": "#06c"}),
        ], style={"margin": "8px 0"}),
        html.Div([
            dcc.Checklist(id="rangeon",
                          options=[{"label": " 範囲選択(一部だけ表示・保存)",
                                    "value": "on"}], value=[],
                          style={"fontWeight": "bold"}),
            html.Label("左右 X (mm)"),
            _rslider("rx", ext["x"][0], ext["x"][1]),
            html.Label("上下 Y (mm)"),
            _rslider("ry", ext["y"][0], ext["y"][1]),
            html.Label("距離 Z (mm)"),
            _rslider("rz", ext["z"][0], ext["z"][1]),
        ], style={"border": "1px solid #ccc", "padding": "8px",
                  "margin": "8px 0"}),
        dcc.Graph(id="g3d", config={"scrollZoom": False}),
        html.P("マウスでドラッグ=回転。拡大縮小はトラックパッドのピンチ、または"
               "右上の拡大ボタン。ページは普通にスクロールできます。"
               "終了はこのウインドウでControl+C。", style={"color": "#555"}),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    @app.callback(Output("g3d", "figure"),
                  Input("fr", "value"), Input("maxd", "value"),
                  Input("psz", "value"), Input("colorby", "value"),
                  Input("resetbtn", "n_clicks"), Input("rangeon", "value"),
                  Input("rx", "value"), Input("ry", "value"),
                  Input("rz", "value"))
    def _update(fr, maxd, psz, colorby, reset, rangeon, rx, ry, rz):
        fr = int(fr) if fr is not None else 0
        if S["is_dat"]:
            fr = min(max(0, fr), max(0, S["n"] - 1))
        md = None if (maxd is None or maxd >= 4000) else float(maxd)
        pc = load(S["path"], fr, S["is_dat"], md, 40000)
        return fig3d(pc, int(psz), colorby or "z", _box(rangeon, rx, ry, rz))

    @app.callback(Output("dl", "data"), Input("csvbtn", "n_clicks"),
                  State("fr", "value"), State("maxd", "value"),
                  State("rangeon", "value"), State("rx", "value"),
                  State("ry", "value"), State("rz", "value"),
                  prevent_initial_call=True)
    def _csv(nclk, fr, maxd, rangeon, rx, ry, rz):
        import io
        fr = int(fr) if fr is not None else 0
        if S["is_dat"]:
            fr = min(max(0, fr), max(0, S["n"] - 1))
        md = None if (maxd is None or maxd >= 4000) else float(maxd)
        pc = load(S["path"], fr, S["is_dat"], md, 200000)
        if pc is None or len(pc.xyz) == 0:
            raise dash.exceptions.PreventUpdate
        xyz, inten = _clip(pc)
        xyz, inten = _apply_box(xyz, inten, _box(rangeon, rx, ry, rz))
        buf = io.StringIO()
        np.savetxt(buf, np.column_stack([xyz, inten]), delimiter=",",
                   header="x,y,z,intensity", comments="", fmt="%.2f")
        base = os.path.splitext(os.path.basename(S["path"]))[0]
        name = f"{base}_frame{fr}.csv" if S["is_dat"] else f"{base}.csv"
        return {"content": buf.getvalue(), "filename": name}

    @app.callback(Output("dl", "data", allow_duplicate=True),
                  Output("recmsg", "children"),
                  Input("recbtn", "n_clicks"),
                  State("fr", "value"), State("maxd", "value"),
                  State("colorby", "value"), State("rangeon", "value"),
                  State("rx", "value"), State("ry", "value"),
                  State("rz", "value"), prevent_initial_call=True)
    def _record(nclk, fr, maxd, colorby, rangeon, rx, ry, rz):
        fr = int(fr) if fr is not None else 0
        if S["is_dat"]:
            fr = min(max(0, fr), max(0, S["n"] - 1))
        md = None if (maxd is None or maxd >= 4000) else float(maxd)
        pc = load(S["path"], fr, S["is_dat"], md, 40000)
        if pc is None or len(pc.xyz) == 0:
            return dash.no_update, "表示する点がありません。"
        xyz, inten = _clip(pc)
        xyz, inten = _apply_box(xyz, inten, _box(rangeon, rx, ry, rz))
        try:
            b64 = _make_gif(xyz, inten, colorby or "z")
        except Exception as e:
            return dash.no_update, f"録画に失敗しました: {e}"
        base = os.path.splitext(os.path.basename(S["path"]))[0]
        return ({"content": b64, "filename": f"{base}_frame{fr}.gif",
                 "base64": True}, "保存しました(回転GIF)。")

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
                "現在のデータ: " + data_label(p))

    @app.callback(
        Output("dd", "options", allow_duplicate=True),
        Output("dd", "value", allow_duplicate=True),
        Output("scanmsg", "children", allow_duplicate=True),
        Input("vol", "value"), prevent_initial_call=True)
    def _pickvol(volpath):
        if not volpath:
            raise dash.exceptions.PreventUpdate
        files = scan_folder(volpath)

        def _mt(f):
            try:
                return os.path.getmtime(f)
            except OSError:
                return 0.0
        files.sort(key=_mt, reverse=True)
        nm = os.path.basename(volpath)
        if not files:
            return [], None, f"『{nm}』にデータ(.dat/.csv)が見つかりません。"
        return (_opts(files), files[0],
                f"『{nm}』から {len(files)} 件。②でデータを選んでください。")

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
    print(f"  ★ブラウザが開かないときは、Chromeのアドレス欄に {url} を入れてください")
    _write_url_file(url)
    if not os.environ.get("TOFVIZ_NO_AUTOOPEN"):
        _open_browser(url)
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except AttributeError:
        app.run_server(host="0.0.0.0", port=port, debug=False)
