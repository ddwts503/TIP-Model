"""ToForge .dat のコマ送りプレビュー(顔のコマ探し用).

距離ヒートマップをスライダーでパラパラ送り、良いフレーム番号を見つける。
見つけた番号を `--frame N` に使って 3D 表示・輪切りする。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import toforge


def browse_frames(
    path: str,
    *,
    start_frame: Optional[int] = None,
    min_depth: float = toforge.MIN_DEPTH_DEFAULT,
    max_depth: Optional[float] = None,
    cmap: str = "turbo",
):
    """スライダーでフレームを送りながら距離ヒートマップを表示する(GUI必須)。"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    n = toforge.count_frames(path)
    if n == 0:
        raise ValueError("ToForge形式として読めませんでした。")
    cur = n // 2 if start_frame is None else max(0, min(start_frame, n - 1))

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")  # 無効画素は黒

    img0 = toforge.read_depth_image(path, cur, min_depth=min_depth,
                                    max_depth=max_depth)
    im = ax.imshow(img0, cmap=cmap_obj, origin="upper")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="distance Z [mm]")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    def render(frame):
        img = toforge.read_depth_image(path, frame, min_depth=min_depth,
                                       max_depth=max_depth)
        im.set_data(img)
        finite = np.isfinite(img)
        if finite.any():
            im.set_clim(np.nanmin(img), np.nanmax(img))
        title.set_text(f"frame {frame} / {n - 1}    "
                       f"(use:  --frame {frame})")
        fig.canvas.draw_idle()

    # フレーム送りスライダー
    sax = plt.axes((0.15, 0.10, 0.7, 0.03))
    slider = Slider(sax, "frame", 0, n - 1, valinit=cur, valstep=1)
    slider.on_changed(lambda v: render(int(v)))

    # 前後ボタン(微調整用)
    def step(delta):
        slider.set_val(int(np.clip(slider.val + delta, 0, n - 1)))

    bax_prev = plt.axes((0.15, 0.03, 0.12, 0.05))
    bax_next = plt.axes((0.73, 0.03, 0.12, 0.05))
    b_prev = Button(bax_prev, "◀ -10")
    b_next = Button(bax_next, "+10 ▶")
    b_prev.on_clicked(lambda e: step(-10))
    b_next.on_clicked(lambda e: step(+10))

    # キーボード左右でも送れる
    def on_key(event):
        if event.key == "right":
            step(+1)
        elif event.key == "left":
            step(-1)
        elif event.key == "up":
            step(+10)
        elif event.key == "down":
            step(-10)

    fig.canvas.mpl_connect("key_press_event", on_key)

    render(cur)
    print(f"[browse] {n:,} フレーム。良いコマの番号を見つけたら、"
          f"その番号を --frame に使ってください。")
    plt.show()
    return slider
