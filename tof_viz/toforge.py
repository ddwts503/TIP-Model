"""ATR-Promotions「TOFORGE」計測ソフトが保存する 3D ToF .dat の読み込み.

ToForge.dll を逆コンパイルして解読した仕様に基づく(format 006)。

ファイル構造:
  [フレーム0][フレーム1]...
  各フレーム = 256バイトのASCIIヘッダ + 640*480画素データ
  1画素 = 8バイト = 4つの16bit値 (X, Y, Z, IR) のリトルエンディアン
  変換:
    X = (uint16 - 32768) * 0.25   [mm]
    Y = (uint16 - 32768) * 0.25   [mm]
    Z =  uint16          * 0.25   [mm]   (距離。0は無効)
    IR = int16                            (明るさ)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .loader import PointCloud

# --- ToForge.dll の定数より ---
HEADER_SIZE = 256
SENSOR_WIDTH = 640
SENSOR_HEIGHT = 480
NUM_CHANNELS = 4               # X, Y, Z, IR
BYTES_PER_VALUE = 2            # int16
FRAME_DATA_BYTES = SENSOR_WIDTH * SENSOR_HEIGHT * NUM_CHANNELS * BYTES_PER_VALUE  # 2,457,600
FRAME_STRIDE = HEADER_SIZE + FRAME_DATA_BYTES                                    # 2,457,856
SCALE = 0.25                   # 0.25 mm / LSB
XY_BIAS = 32768                # X,Y はこのオフセットを引く
MIN_DEPTH_DEFAULT = 500        # ToForge MIN_DEPTH (mm)


def count_frames(path: str) -> int:
    """ファイルに含まれるフレーム数を返す。"""
    size = os.path.getsize(path)
    return size // FRAME_STRIDE


def read_header(path: str, frame_index: int = 0) -> dict:
    """指定フレームの256バイトASCIIヘッダを読み、主要フィールドを返す。"""
    with open(path, "rb") as f:
        f.seek(frame_index * FRAME_STRIDE)
        raw = f.read(HEADER_SIZE)
    text = raw.decode("ascii", errors="replace")
    return {
        "raw": text,
        "marker": text[0:1],          # '*'
        "format_version": text[1:4],  # '006'
    }


def read_depth_image(
    path: str,
    frame_index: int = 0,
    *,
    min_depth: float = MIN_DEPTH_DEFAULT,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """1フレームの距離(Z)を 480x640 の2D配列 [mm] で返す。

    無効画素(範囲外/Z=0)は NaN。コマ送りプレビュー用。
    """
    n = count_frames(path)
    if n == 0:
        raise ValueError("ToForge形式のフレームが見つかりません。")
    frame_index = max(0, min(frame_index, n - 1))
    with open(path, "rb") as f:
        f.seek(frame_index * FRAME_STRIDE + HEADER_SIZE)
        buf = f.read(FRAME_DATA_BYTES)
    raw = np.frombuffer(buf, dtype="<u2").reshape(SENSOR_HEIGHT, SENSOR_WIDTH,
                                                  NUM_CHANNELS)
    z = raw[:, :, 2].astype(np.float64) * SCALE
    mask = z >= float(min_depth)
    if max_depth is not None:
        mask &= z <= float(max_depth)
    z = np.where(mask, z, np.nan)
    return z


def read_frame(
    path: str,
    frame_index: int = 0,
    *,
    min_depth: float = MIN_DEPTH_DEFAULT,
    max_depth: Optional[float] = None,
    max_points: Optional[int] = None,
) -> PointCloud:
    """1フレームを点群 (PointCloud) として読み込む。

    min_depth / max_depth (mm) で距離をフィルタ。無効画素(Z=0)は除外。
    巨大ファイルでも1フレーム(約2.4MB)しか読まないのでメモリ安全。
    """
    n = count_frames(path)
    if n == 0:
        raise ValueError("ToForge形式のフレームが見つかりません(ファイルが小さすぎます)。")
    if frame_index < 0:
        frame_index += n
    if not (0 <= frame_index < n):
        raise IndexError(f"frame_index は 0〜{n-1} の範囲で指定してください(指定値 {frame_index})。")

    with open(path, "rb") as f:
        f.seek(frame_index * FRAME_STRIDE + HEADER_SIZE)
        buf = f.read(FRAME_DATA_BYTES)
    if len(buf) < FRAME_DATA_BYTES:
        raise ValueError("フレームデータが途中で切れています。")

    # (H, W, 4) の int16 / uint16 として解釈
    raw = np.frombuffer(buf, dtype="<u2").reshape(SENSOR_HEIGHT, SENSOR_WIDTH, NUM_CHANNELS)

    xch = raw[:, :, 0].astype(np.float64)
    ych = raw[:, :, 1].astype(np.float64)
    zch = raw[:, :, 2].astype(np.float64)
    irch = raw[:, :, 3].astype(np.int16)  # IR は符号付き

    x = (xch - XY_BIAS) * SCALE
    y = (ych - XY_BIAS) * SCALE
    z = zch * SCALE
    ir = irch.astype(np.float64)

    # 有効画素マスク: 距離が範囲内
    mask = z >= float(min_depth)
    if max_depth is not None:
        mask &= z <= float(max_depth)

    xyz = np.column_stack([x[mask], y[mask], z[mask]])
    intensity = ir[mask]

    if len(xyz) == 0:
        raise ValueError(
            f"距離 {min_depth}〜{max_depth} mm に有効な点がありません。"
            "--min-depth / --max-depth を調整してください。"
        )

    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        intensity = intensity[idx]

    src = f"{path} [frame {frame_index}/{n-1}]"
    return PointCloud(xyz=xyz, intensity=intensity, source=src)
