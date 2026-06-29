"""ビフォー/アフターを比べて「小顔になったか」を計測するブラウザアプリ.

2つの静止ToFデータ(.dat のフレーム/整列CSV)をベッド基準に整列し、
ユーザが画像上で顔の中心をクリック→その周りを顔として切り出し、
顔の幅・断面周囲・断面面積・体積を前後で比較する。位置合わせ(ICP)も可。
"""
from __future__ import annotations

import glob
import os
import webbrowser
from functools import lru_cache
from typing import Optional

import numpy as np

from .bed import compute_bed_aligned
from .loader import PointCloud, load_points
from .measure import measure_section


DATA_EXTS = (".dat", ".csv")


def _walk_collect(root, *, maxdepth=None, min_size=1_000_000):
    """root配下の .dat/.csv を集める(大文字小文字を区別しない・エラーに強い)。"""
    out = []
    root = root.rstrip("/")
    if not os.path.isdir(root):
        return out
    base = root.count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        # 隠し/システムフォルダ(.Spotlight等やTime Machine)は除外
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d != "Backups.backupdb"]
        if maxdepth is not None and (dirpath.count(os.sep) - base) >= maxdepth:
            dirnames[:] = []
        for fn in filenames:
            if fn.startswith("."):
                continue
            if fn.lower().endswith(DATA_EXTS):
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.isfile(p) and os.path.getsize(p) >= min_size:
                        out.append(p)
                except OSError:
                    pass
    return out


def find_data_files(extra=(), *, deep=False):
    """Mac内の計測データ候補(.dat / .csv)を探す。SSD・デスクトップ等を走査。

    deep=False(既定)は浅め(深さ3)で高速=起動時用。deep=True は奥まで
    くまなく(時間がかかる)=「再スキャン」用。
    """
    found = []
    # /Volumes 配下の各SSDを個別に走査(1台がエラーでも他は拾える)
    vol = "/Volumes"
    vdepth = None if deep else 3
    if os.path.isdir(vol):
        try:
            entries = [os.path.join(vol, d) for d in os.listdir(vol)]
        except OSError:
            entries = []
        for d in entries:
            try:
                found += _walk_collect(d, maxdepth=vdepth)
            except Exception:
                pass
    # ホーム配下は重くならないよう浅めに(深さ4まで)
    for r in (os.path.expanduser("~/Desktop"),
              os.path.expanduser("~/Downloads"),
              os.path.expanduser("~/Documents")):
        found += _walk_collect(r, maxdepth=4)
    # 現在使用中のファイルも必ず候補に入れ、重複を除く
    files = list(dict.fromkeys([p for p in extra if p] + found))

    def _mtime(f):
        try:
            return os.path.getmtime(f)
        except OSError:
            return 0.0
    files.sort(key=_mtime, reverse=True)
    return files


def list_volumes():
    """マウント中のSSD等(/Volumes 配下のドライブ)のパス一覧。"""
    out = []
    try:
        for d in sorted(os.listdir("/Volumes")):
            p = os.path.join("/Volumes", d)
            if os.path.isdir(p) and not d.startswith("."):
                out.append(p)
    except OSError:
        pass
    return out


def data_label(p):
    """ドロップダウンの表示名。先頭に【SSD名】、末尾にサイズを出して見分けやすく。"""
    name = os.path.basename(p)
    parts = p.split(os.sep)
    if len(parts) >= 3 and parts[1] == "Volumes":
        vol = parts[2]                       # SSD(ボリューム)名
        head = f"🟦【{vol}】"
    else:
        head = "💻【Mac本体】"
    try:
        gb = os.path.getsize(p) / 1e9
        size = f"  ・{gb:.1f}GB"
    except OSError:
        size = ""
    return f"{head} {name}{size}"


def scan_folder(path):
    """指定したファイル/フォルダから .dat・.csv を集める(フォルダは再帰)。"""
    if not path:
        return []
    path = os.path.expanduser(path.strip()).rstrip("/")
    if os.path.isfile(path):
        return [path]
    out = []
    if os.path.isdir(path):
        for ext in ("dat", "csv"):
            out += glob.glob(os.path.join(path, "**", "*." + ext),
                             recursive=True)
    return [f for f in out if os.path.isfile(f)]


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
    """鼻=中央付近(|x|<400で腕を除外)の一番手前(高さ最大)を自動検出。"""
    if len(zp) == 0:
        return (0.0, 0.0)
    c = np.abs(xp) < 400
    if c.sum() < 20:
        c = np.ones(len(xp), bool)
    xc, yc, zc = xp[c], yp[c], zp[c]
    m = zc >= np.percentile(zc, 98)
    return (float(np.median(xc[m])), float(np.median(yc[m])))


def _roll(x, y, cx, cy, deg):
    """画面内(x-y)で中心(cx,cy)まわりに回転=体軸の傾き補正。"""
    if not deg or len(x) == 0:
        return x, y
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    dx, dy = x - cx, y - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def _yaw(x, y, h, cx, deg):
    """中心軸(x=cx の縦軸)まわりに左右回転(頭の向き直し)。x と 高さ h を回す。"""
    if not deg or len(x) == 0:
        return x, y, h
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    ch = float(np.median(h))
    dx = x - cx
    dh = h - ch
    xn = cx + dx * c - dh * s
    hn = ch + dx * s + dh * c
    return xn, y, hn


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


def face_metrics(xp, yp, zp, *, axis, position, thickness=10.0):
    pc = PointCloud(xyz=np.column_stack([xp, yp, zp]))
    try:
        length, area, curve, _ = measure_section(
            pc, axis=axis, position=position, thickness=thickness)
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

    # 実行中に差し替え可能な現在のデータ状態(ドロップダウンで変更)
    S = {"before": before_path, "after": after_path,
         "b_is_dat": b_is_dat, "a_is_dat": a_is_dat, "nB": nB, "nA": nA}

    @lru_cache(maxsize=128)
    def load_align(path, frame, is_dat, rot=0.0):
        """カメラそのままの向き(顔は正立)。人物だけ残し、鼻=手前を高さに。

        rot[度]だけ画面内で回転(傾き補正)。
        """
        try:
            if is_dat:
                from . import toforge
                pc = toforge.read_frame(path, frame, min_depth=300.0)
            else:
                pc = load_points(path)
            x, y, z = np.asarray(pc.x), np.asarray(pc.y), np.asarray(pc.z)
        except Exception:
            # 人が写っていない/有効な点が無いコマでも落とさず空で返す
            e = np.array([])
            return e, e, e
        if len(z) == 0:
            return x, y, z
        znear = float(np.percentile(z, 1))       # 最も手前(鼻側)
        keep = (z >= znear) & (z <= znear + 350)  # 人物だけ(背景/ベッド除去)
        x, y, z = x[keep], y[keep], z[keep]
        height = float(z.max()) - z               # 手前(鼻)ほど大きい高さ
        if rot:
            th = np.radians(rot)
            c, s = np.cos(th), np.sin(th)
            x, y = x * c - y * s, x * s + y * c
        return x, y, height

    def _good_frame(path, is_dat, n, half):
        """人がよく写っているコマを自動で選ぶ。half='first'は前半,'second'は後半。"""
        if not is_dat or n <= 1:
            return 0
        if half == "first":
            cand = np.linspace(0, max(1, n // 2), 8)
        else:
            cand = np.linspace(n // 2, n - 1, 8)
        best, best_cnt = int(cand[0]), -1
        for f in np.unique(cand.astype(int)):
            _, _, h = load_align(path, int(f), is_dat)
            cnt = int(_subject_mask(h).sum()) if len(h) else 0
            if cnt > best_cnt:
                best_cnt, best = cnt, int(f)
        return best

    fB0 = (frame_before if frame_before is not None
           else _good_frame(before_path, b_is_dat, nB, "first"))
    fA0 = (frame_after if frame_after is not None
           else _good_frame(after_path, a_is_dat, nA, "second"))

    def topfig(xp, yp, zp, title, center, radius, sdir=None, slicepos=None,
               zoom=False):
        m = _subject_mask(zp)
        fig = go.Figure(go.Scattergl(
            x=xp[m], y=yp[m], mode="markers",
            marker=dict(size=3, color=zp[m], colorscale="Turbo", opacity=0.55)))
        cx, cy = center
        # shape[0]=ドラッグで動かせる円(赤・半透明の塗り=内側をつかんで移動)
        fig.add_shape(type="circle", x0=cx - radius, y0=cy - radius,
                      x1=cx + radius, y1=cy + radius, layer="above",
                      line=dict(color="red", width=3),
                      fillcolor="rgba(255,0,0,0.12)")
        # shape[1]=ドラッグで動かせるスライス線(緑・太め)
        if sdir is not None and m.sum() > 0:
            xs0, xs1 = float(xp[m].min()), float(xp[m].max())
            ys0, ys1 = float(yp[m].min()), float(yp[m].max())
            if sdir == "vert":      # 縦スライス=縦線
                px = slicepos if slicepos is not None else cx
                fig.add_shape(type="line", x0=px, x1=px, y0=ys0, y1=ys1,
                              line=dict(color="lime", width=1), layer="above")
            else:                   # 横スライス=横線
                py = slicepos if slicepos is not None else cy
                fig.add_shape(type="line", x0=xs0, x1=xs1, y0=py, y1=py,
                              line=dict(color="lime", width=1), layer="above")
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                                 marker=dict(color="red", size=14, symbol="x",
                                             line=dict(color="white", width=1))))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title=title, height=380, showlegend=False,
                          margin=dict(l=20, r=10, t=40, b=20))
        if zoom:                       # 顔を中心に拡大表示
            pad = radius * 1.8
            fig.update_xaxes(range=[cx - pad, cx + pad])
            fig.update_yaxes(range=[cy - pad, cy + pad])
        if m.sum() == 0:
            fig.add_annotation(text="このコマには人が写っていません。<br>"
                               "コマのスライダーを動かしてください。",
                               showarrow=False, font=dict(size=16, color="red"),
                               xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    def compute(fB, fA, posB, posA, radius, register, cB, cA, sdir,
                byaw=0.0, ayaw=0.0, broll=0.0, aroll=0.0, thick=5.0):
        bx, by, bz = load_align(S["before"], fB, S["b_is_dat"])
        ax, ay, az = load_align(S["after"], fA, S["a_is_dat"])
        # ビフォー/アフター別々に: 傾き(軸回転 roll)→ 左右向き(yaw)
        bx, by = _roll(bx, by, cB[0], cB[1], broll)
        ax, ay = _roll(ax, ay, cA[0], cA[1], aroll)
        bx, by, bz = _yaw(bx, by, bz, cB[0], byaw)
        ax, ay, az = _yaw(ax, ay, az, cA[0], ayaw)
        bx, by, bz, cB = _crop(bx, by, bz, cB, radius)
        ax, ay, az, cA = _crop(ax, ay, az, cA, radius)
        # 高さを「顔の縁(下から5%)」基準に各自そろえる→前後の突出を公平に比較
        if len(bz):
            bz = bz - np.percentile(bz, 5)
        if len(az):
            az = az - np.percentile(az, 5)
        if register and len(bz) > 20 and len(az) > 20:
            reg = register_icp(np.column_stack([ax, ay, az]),
                               np.column_stack([bx, by, bz]))
            ax, ay, az = reg[:, 0], reg[:, 1], reg[:, 2]
        baseB = float(np.percentile(bz, 10)) if len(bz) else 0.0
        baseA = float(np.percentile(az, 10)) if len(az) else 0.0
        volB = volume_face(bx, by, bz, baseB)
        volA = volume_face(ax, ay, az, baseA)
        # スライス位置(絶対座標)はビフォー/アフター別々。未指定なら各円の中心
        ci = 0 if sdir == "vert" else 1
        axis = "x" if sdir == "vert" else "y"
        pB = posB if posB is not None else cB[ci]
        pA = posA if posA is not None else cA[ci]
        mB = face_metrics(bx, by, bz, axis=axis, position=pB, thickness=thick)
        mA = face_metrics(ax, ay, az, axis=axis, position=pA, thickness=thick)
        return (bx, by, bz, ax, ay, az, mB, mA, volB, volA, axis)

    def make_outputs(state):
        (bx, by, bz, ax, ay, az, mB, mA, volB, volA, axis) = state
        # 左右(横方向)を各曲線の中心でそろえて重ねる(高さはそのまま)
        def _xc(curve):
            if len(curve) == 0:
                return curve
            cc = (curve[:, 0].min() + curve[:, 0].max()) / 2
            out = curve.copy()
            out[:, 0] = out[:, 0] - cc
            return out
        cb_c = _xc(mB["curve"])
        ca_c = _xc(mA["curve"])
        secf = go.Figure()
        if len(cb_c):
            secf.add_trace(go.Scatter(x=cb_c[:, 0], y=cb_c[:, 1],
                           mode="lines+markers", name="ビフォー", line_color="blue"))
        if len(ca_c):
            secf.add_trace(go.Scatter(x=ca_c[:, 0], y=ca_c[:, 1],
                           mode="lines+markers", name="アフター", line_color="red"))
        secf.update_yaxes(scaleanchor="x", scaleratio=1)
        if axis == "x":   # 縦スライス: 横顔プロフィール(上下 × 突出)
            title = "縦スライス = 横顔プロフィール"
            xlab, ylab = "上下 [mm]", "突出(高さ)[mm]"
        else:             # 横スライス: 左右の断面(左右 × 突出)
            title = "横スライス = 左右の断面"
            xlab, ylab = "左右 [mm]", "突出(高さ)[mm]"
        secf.update_layout(title=title + "  青=ビフォー 赤=アフター", height=380,
                           xaxis_title=xlab, yaxis_title=ylab)
        wB, lB, dB = face_dims(bx, by, bz)
        wA, lA, dA = face_dims(ax, ay, az)
        rB = dB / wB if wB else 0.0     # 立体度(高さ÷幅)
        rA = dA / wA if wA else 0.0

        def trow(name, b, a, unit, key=False, good="down"):
            d = a - b
            pct = (d / b * 100) if b else 0
            arrow = "⬇" if d < 0 else ("⬆" if d > 0 else "→")
            style = {"fontWeight": "bold"} if key else {}
            return html.Tr([html.Td(name, style=style),
                            html.Td(f"{b:.1f}{unit}"), html.Td(f"{a:.1f}{unit}"),
                            html.Td(f"{d:+.1f} ({pct:+.1f}%){arrow}", style=style)])
        table = html.Table([
            html.Thead(html.Tr([html.Th("項目"), html.Th("ビフォー"), html.Th("アフター"),
                                html.Th("変化")])),
            html.Tbody([
                trow("◎ 顔幅(左右)→減れば小顔", wB, wA, "mm", key=True),
                trow("◎ 高さ(突出)→増えればリフト", dB, dA, "mm", key=True),
                trow("◎ 立体度(高さ÷幅)→増で引締", rB, rA, "", key=True),
                trow("顔の長さ(頭足)", lB, lA, "mm"),
                trow("(参考)体積 ※骨格不変なら減らない", volB, volA, "cm³"),
                trow("(参考)断面の面積", mB["area"]/100, mA["area"]/100, "cm²")])],
            style={"fontSize": "16px"})
        # 小顔 = 幅が減り、高さ(突出)が増える/立体度が上がる
        narrower = wA < wB - 0.5
        taller = dA > dB + 0.5
        solid_up = rA > rB
        if narrower and (taller or solid_up):
            verdict = "✅ 小顔(引き締まり・リフトアップ)= 幅が減り立体的に"
        elif narrower:
            verdict = "○ 幅は減少(高さは横ばい)"
        elif wA > wB + 0.5:
            verdict = "❌ 顔幅が増えています"
        else:
            verdict = "△ ほぼ変化なし"
        return secf, table, verdict

    # --- 顔の初期中心(被写体の重心)とスライダー範囲を決める ---
    bxf, byf, bzf = load_align(before_path, fB0, b_is_dat)
    axf, ayf, azf = load_align(after_path, fA0, a_is_dat)

    def _face0(x, y, h):
        """初期円=中央寄り(|x|<400)で一番手前(高さ最大)=顔の鼻。腕は除く。"""
        if len(h) == 0:
            return 0.0, 0.0
        c = np.abs(x) < 400
        if c.sum() < 20:
            c = np.ones(len(x), bool)
        xc, yc, hc = x[c], y[c], h[c]
        m = hc >= np.percentile(hc, 98)
        return float(np.median(xc[m])), float(np.median(yc[m]))
    dbx, dby = _face0(bxf, byf, bzf)
    dax, day = _face0(axf, ayf, azf)
    # スライダー範囲は人物の点から(±1500に制限)
    allx = np.concatenate([bxf, axf]) if len(bxf) and len(axf) else np.array([0.0])
    ally = np.concatenate([byf, ayf]) if len(byf) and len(ayf) else np.array([0.0])
    xlo = max(-1500.0, float(np.percentile(allx, 1)) - 100)
    xhi = min(1500.0, float(np.percentile(allx, 99)) + 100)
    ylo = max(-1500.0, float(np.percentile(ally, 1)) - 100)
    yhi = min(1500.0, float(np.percentile(ally, 99)) + 100)

    app = dash.Dash(__name__)

    frame_ctrls = [
        html.Label("ビフォーのコマ", id="fblab"),
        dcc.Slider(0, max(0, nB - 1), max(1, (nB // 200) or 1), value=fB0,
                   id="fb", tooltip={"placement": "bottom",
                                     "always_visible": True}),
        html.Label("アフターのコマ", id="falab"),
        dcc.Slider(0, max(0, nA - 1), max(1, (nA // 200) or 1), value=fA0,
                   id="fa", tooltip={"placement": "bottom",
                                     "always_visible": True})]

    def cslider(id_, lo, hi, val):
        return dcc.Slider(round(lo), round(hi), 5, value=round(val), id=id_,
                          tooltip={"placement": "bottom", "always_visible": True})

    editcfg = {"editable": True, "edits": {"shapePosition": True},
               "scrollZoom": True}

    def _opts(paths):
        return [{"label": data_label(p), "value": p} for p in paths]

    dat_opts = _opts(find_data_files(extra=[before_path, after_path]))

    app.layout = html.Div([
        html.H2("ビフォー・アフター 小顔チェック  [版 v31]"),
        html.Div([
            html.B("データの選択(別のファイルに変えられます)"),
            html.Label("① SSD(ドライブ)を選ぶ"),
            dcc.Dropdown(id="vol",
                         options=[{"label": "🟦 " + os.path.basename(p),
                                   "value": p} for p in list_volumes()],
                         placeholder="つないでいるSSDを選ぶと、中のデータが下に出ます",
                         clearable=False),
            html.Label("② 計測データ(.dat)を選ぶ"),
            dcc.Dropdown(id="dd", options=dat_opts, value=before_path,
                         clearable=False),
            html.Button("▶ このデータで読み込む", id="loadbtn", n_clicks=0,
                        style={"fontSize": "16px", "marginTop": "8px",
                               "padding": "6px 14px"}),
            html.Div(id="loadmsg",
                     children="現在のデータ: " + data_label(before_path),
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
        html.P("赤い円の内側をドラッグして顔へ。緑の線(スライス位置)もドラッグで"
               "動かせます。左=ビフォー、右=アフター。"),
        html.Div([
            dcc.Graph(id="gb", config=editcfg,
                      style={"display": "inline-block", "width": "49%"}),
            dcc.Graph(id="ga", config=editcfg,
                      style={"display": "inline-block", "width": "49%"})]),
        dcc.Store(id="cb", data=[dbx, dby]),
        dcc.Store(id="ca", data=[dax, day]),
        dcc.Store(id="hposB", data=None),
        dcc.Store(id="hposA", data=None),
        html.Div(frame_ctrls),
        html.Div(dcc.Checklist(id="reg",
                 options=[{"label": "  ✔ 2つの顔を自動で重ねて比較する"
                           "(手動で合わせる時はOFFのまま)", "value": "on"}],
                 value=[], style={"fontSize": "18px"}),
                 style={"border": "2px solid #888", "padding": "10px",
                        "margin": "10px 0", "background": "#f4f4f4"}),
        html.B("傾き補正(軸回転・ビフォー/アフター別々)"),
        html.Label("ビフォー 傾き(度)"),
        dcc.Slider(-180, 180, 1, value=0, id="broll",
                   marks={-180: "-180", -90: "-90", 0: "0", 90: "90",
                          180: "180"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("アフター 傾き(度)"),
        dcc.Slider(-180, 180, 1, value=0, id="aroll",
                   marks={-180: "-180", -90: "-90", 0: "0", 90: "90",
                          180: "180"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.B("顔の左右向き補正(中心軸まわり・別々)"),
        html.Label("ビフォー 左右回転(度)"),
        dcc.Slider(-45, 45, 1, value=0, id="byaw",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("アフター 左右回転(度)"),
        dcc.Slider(-45, 45, 1, value=0, id="ayaw",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.B("円の位置の微調整(1mm刻み・ドラッグの後の細かい合わせ)"),
        html.Label("ビフォー 円 左右 / 上下"),
        dcc.Slider(-60, 60, 1, value=0, id="bfx",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Slider(-60, 60, 1, value=0, id="bfy",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("アフター 円 左右 / 上下"),
        dcc.Slider(-60, 60, 1, value=0, id="afx",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Slider(-60, 60, 1, value=0, id="afy",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("顔の範囲(半径 mm)= 円の大きさ(顔だけに小さく)"),
        dcc.Slider(30, 300, 1, value=85, id="r",
                   marks={30: "30", 100: "100", 200: "200", 300: "300"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Checklist(id="zoomface",
                      options=[{"label": "  🔍 顔をアップで表示(円のまわりを拡大)",
                                "value": "on"}],
                      value=[], style={"fontSize": "16px"}),
        html.Label("断面の向き(切り替えるとスライス線の位置はリセット)"),
        dcc.RadioItems(id="sdir", value="horiz", inline=True,
                       options=[{"label": " 横スライス(左右の断面)", "value": "horiz"},
                                {"label": " 縦スライス(横顔プロフィール)", "value": "vert"}],
                       style={"fontSize": "16px"}),
        html.Label("断面の厚み(mm)"),
        dcc.Slider(1, 20, 1, value=5, id="thick",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("ビフォー 断面位置の微調整(中心から mm・0.5mm刻み)"),
        dcc.Slider(-150, 150, 0.5, value=0, id="sB",
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("アフター 断面位置の微調整(中心から mm・0.5mm刻み)"),
        dcc.Slider(-150, 150, 0.5, value=0, id="sA",
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="sec"),
        html.Div(id="tbl"),
        html.H3(id="ver"),
        html.P("円を顔にドラッグ→範囲/深さを調整→表と判定を確認。終了はControl+C。"),
    ], style={"fontFamily": "sans-serif", "margin": "20px"})

    def _from_relayout(rl, fallback):
        if rl and "shapes[0].x0" in rl:
            try:
                return [(rl["shapes[0].x0"] + rl["shapes[0].x1"]) / 2,
                        (rl["shapes[0].y0"] + rl["shapes[0].y1"]) / 2]
            except Exception:
                pass
        return fallback

    @app.callback(Output("cb", "data"), Input("gb", "relayoutData"),
                  State("cb", "data"), State("bfx", "value"),
                  State("bfy", "value"), prevent_initial_call=True)
    def _cb(rl, cur, bfx, bfy):
        c = _from_relayout(rl, None)
        if c is None:
            return cur
        # 表示中心 = 保存中心 + 微調整。ドラッグ後も二重加算でズレないよう微調整を引く
        return [c[0] - float(bfx or 0), c[1] - float(bfy or 0)]

    @app.callback(Output("ca", "data"), Input("ga", "relayoutData"),
                  State("ca", "data"), State("afx", "value"),
                  State("afy", "value"), prevent_initial_call=True)
    def _ca(rl, cur, afx, afy):
        c = _from_relayout(rl, None)
        if c is None:
            return cur
        return [c[0] - float(afx or 0), c[1] - float(afy or 0)]

    def _line_pos(rl, sdir):
        if not rl:
            return None
        if sdir == "vert" and "shapes[1].x0" in rl:
            return (rl["shapes[1].x0"] + rl["shapes[1].x1"]) / 2
        if sdir != "vert" and "shapes[1].y0" in rl:
            return (rl["shapes[1].y0"] + rl["shapes[1].y1"]) / 2
        return None

    # 緑のスライス線: ドラッグ or 1mm刻みスライダー(中心からのオフセット)。別々
    @app.callback(Output("hposB", "data"), Input("gb", "relayoutData"),
                  Input("sB", "value"), Input("sdir", "value"),
                  State("hposB", "data"), State("cb", "data"),
                  prevent_initial_call=True)
    def _hposB(rlb, soff, sdir, cur, cb):
        tid = (dash.callback_context.triggered[0]["prop_id"]
               if dash.callback_context.triggered else "")
        if tid.startswith("sdir"):
            return None
        if tid.startswith("gb"):
            p = _line_pos(rlb, sdir)
            return p if p is not None else cur
        ci = 0 if sdir == "vert" else 1
        c = float(cb[ci]) if cb else 0.0
        return c + float(soff)

    @app.callback(Output("hposA", "data"), Input("ga", "relayoutData"),
                  Input("sA", "value"), Input("sdir", "value"),
                  State("hposA", "data"), State("ca", "data"),
                  prevent_initial_call=True)
    def _hposA(rla, soff, sdir, cur, ca):
        tid = (dash.callback_context.triggered[0]["prop_id"]
               if dash.callback_context.triggered else "")
        if tid.startswith("sdir"):
            return None
        if tid.startswith("ga"):
            p = _line_pos(rla, sdir)
            return p if p is not None else cur
        ci = 0 if sdir == "vert" else 1
        c = float(ca[ci]) if ca else 0.0
        return c + float(soff)

    outs = [Output("gb", "figure"), Output("ga", "figure"),
            Output("sec", "figure"), Output("tbl", "children"),
            Output("ver", "children")]
    ins = [Input("r", "value"), Input("hposB", "data"), Input("hposA", "data"),
           Input("reg", "value"), Input("cb", "data"), Input("ca", "data"),
           Input("sdir", "value"), Input("byaw", "value"), Input("ayaw", "value"),
           Input("broll", "value"), Input("aroll", "value"),
           Input("thick", "value"), Input("bfx", "value"), Input("bfy", "value"),
           Input("afx", "value"), Input("afy", "value"),
           Input("zoomface", "value"),
           Input("fb", "value"), Input("fa", "value")]

    @app.callback(*outs, *ins)
    def _update(r, hposB, hposA, reg, cb, ca, sdir, byaw, ayaw, broll, aroll,
                thick, bfx, bfy, afx, afy, zoomface, fB, fA):
        fB = int(fB) if fB is not None else 0
        fA = int(fA) if fA is not None else 0
        fB = min(max(0, fB), max(0, S["nB"] - 1)) if S["b_is_dat"] else 0
        fA = min(max(0, fA), max(0, S["nA"] - 1)) if S["a_is_dat"] else 0
        # 円の中心 = ドラッグ位置 + 微調整(1mm刻み)
        cB = ((float(cb[0]) if cb else dbx) + float(bfx),
              (float(cb[1]) if cb else dby) + float(bfy))
        cA = ((float(ca[0]) if ca else dax) + float(afx),
              (float(ca[1]) if ca else day) + float(afy))
        pB = float(hposB) if hposB is not None else None
        pA = float(hposA) if hposA is not None else None
        st = compute(int(fB), int(fA), pB, pA, float(r), bool(reg), cB, cA,
                     sdir, float(byaw), float(ayaw), float(broll), float(aroll),
                     float(thick))
        bxx, byy, bzz = load_align(S["before"], int(fB), S["b_is_dat"])
        axx, ayy, azz = load_align(S["after"], int(fA), S["a_is_dat"])
        bxx, byy = _roll(bxx, byy, cB[0], cB[1], float(broll))
        axx, ayy = _roll(axx, ayy, cA[0], cA[1], float(aroll))
        bxx, byy, bzz = _yaw(bxx, byy, bzz, cB[0], float(byaw))
        axx, ayy, azz = _yaw(axx, ayy, azz, cA[0], float(ayaw))
        # スライス線の位置(未ドラッグなら各円の中心)。ビフォー/アフター別々
        ci = 0 if sdir == "vert" else 1
        spb = pB if pB is not None else cB[ci]
        spa = pA if pA is not None else cA[ci]
        zoom = bool(zoomface)
        gb = topfig(bxx, byy, bzz, f"ビフォー (コマ {fB})", cB, float(r),
                    sdir=sdir, slicepos=spb, zoom=zoom)
        ga = topfig(axx, ayy, azz, f"アフター (コマ {fA})", cA, float(r),
                    sdir=sdir, slicepos=spa, zoom=zoom)
        secf, table, verdict = make_outputs(st)
        return gb, ga, secf, table, verdict

    # ドロップダウンで選んだ別データを読み込む(コマ範囲・初期円をリセット)
    @app.callback(
        Output("fb", "max"), Output("fb", "value"), Output("fb", "step"),
        Output("fa", "max"), Output("fa", "value"), Output("fa", "step"),
        Output("cb", "data", allow_duplicate=True),
        Output("ca", "data", allow_duplicate=True),
        Output("hposB", "data", allow_duplicate=True),
        Output("hposA", "data", allow_duplicate=True),
        Output("loadmsg", "children"),
        Input("loadbtn", "n_clicks"),
        State("dd", "value"),
        prevent_initial_call=True)
    def _reload(n, path):
        if not path:
            raise dash.exceptions.PreventUpdate
        # 同じ録画をビフォー・アフター両方に使う(コマで前後を選ぶ)
        S["before"] = S["after"] = path
        is_dat = path.lower().endswith(".dat")
        S["b_is_dat"] = S["a_is_dat"] = is_dat
        from . import toforge
        nfr = toforge.count_frames(path) if is_dat else 0
        S["nB"] = S["nA"] = nfr
        nb_max = na_max = max(0, nfr - 1)
        fb0 = _good_frame(path, is_dat, nfr, "first")
        fa0 = _good_frame(path, is_dat, nfr, "second")
        bx, by, bz = load_align(path, fb0, is_dat)
        ax, ay, az = load_align(path, fa0, is_dat)
        nbc = list(_face0(bx, by, bz))
        nac = list(_face0(ax, ay, az))
        msg = "現在のデータ: " + data_label(path)
        return (nb_max, fb0, max(1, (S["nB"] // 200) or 1),
                na_max, fa0, max(1, (S["nA"] // 200) or 1),
                nbc, nac, None, None, msg)

    # ①SSDを選ぶ → その中のデータを一覧にして②に出す(確実に選べる)
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

    # パス貼り付け or 再スキャンで、データ一覧を更新
    @app.callback(
        Output("dd", "options"),
        Output("scanmsg", "children"),
        Input("scanbtn", "n_clicks"), Input("rescanbtn", "n_clicks"),
        State("pathbox", "value"),
        prevent_initial_call=True)
    def _scan(n_scan, n_rescan, pathstr):
        tid = (dash.callback_context.triggered[0]["prop_id"]
               if dash.callback_context.triggered else "")
        deep = tid.startswith("rescanbtn")
        paths = find_data_files(extra=[S["before"], S["after"]], deep=deep)
        msg = f"自動一覧: {len(paths)} 件"
        if tid.startswith("scanbtn"):
            extra = scan_folder(pathstr or "")
            if not extra:
                msg = (f"『{pathstr}』からデータ(.dat/.csv)が見つかりません。"
                       "パスが正しいか確認してください。")
            else:
                msg = f"『{pathstr}』から {len(extra)} 件見つかりました。"
            # 指定パスの結果を先頭に
            paths = list(dict.fromkeys(extra + paths))
        return _opts(paths), msg

    import socket
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break
    url = f"http://127.0.0.1:{port}"

    # 同じWi-Fi上のiPhone等から開けるよう、このMacのLAN IPも表示する
    lan_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))           # 外部に出る経路のIP(送信はしない)
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = None

    print(f"[compare] このMacで開く: {url}")
    if lan_ip and not lan_ip.startswith("127."):
        print(f"[compare] iPhone/iPadで開く(同じWi-Fi): "
              f"http://{lan_ip}:{port}")
        print("  ↑この住所をiPhoneのSafariのアドレス欄に入れてください。")
    print("  顔をクリックして範囲を合わせてください。終了: Control+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except AttributeError:
        app.run_server(host="0.0.0.0", port=port, debug=False)
