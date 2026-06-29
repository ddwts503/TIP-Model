"""ToF CSV/テキストデータの読み込みユーティリティ.

区切り文字(カンマ/タブ/スペース/セミコロン)とヘッダ行を自動判定し、
x, y, z 座標(および任意の強度/intensity 列)を numpy 配列として返す。
"""
from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def saved_and_open(path: str) -> None:
    """保存報告 + 存在確認 + macOS なら自動で開く(全モード共通)。"""
    exists = os.path.exists(path)
    ap = os.path.abspath(path)
    print(f"[saved] {ap}  (exists={exists})")
    if exists and sys.platform == "darwin":
        try:
            subprocess.run(["open", ap], check=False)
        except Exception:
            pass


# ヘッダ行で座標・強度列を探すときの候補名(小文字で比較)
_X_NAMES = ("x", "px", "x[m]", "x_m", "pos_x")
_Y_NAMES = ("y", "py", "y[m]", "y_m", "pos_y")
_Z_NAMES = ("z", "pz", "z[m]", "z_m", "pos_z", "depth", "d")
_I_NAMES = ("intensity", "i", "amplitude", "amp", "reflectance",
            "confidence", "conf", "gray", "value")


@dataclass
class PointCloud:
    """点群データ。xyz は (N, 3)、intensity は (N,) または None。"""

    xyz: np.ndarray
    intensity: Optional[np.ndarray] = None
    source: str = ""

    def __len__(self) -> int:  # noqa: D401
        return int(self.xyz.shape[0])

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]


def _sniff_delimiter(sample: str) -> str:
    """サンプル文字列から区切り文字を推定する。"""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        return dialect.delimiter
    except csv.Error:
        # フォールバック: 出現頻度の高い区切り候補を採用
        counts = {d: sample.count(d) for d in (",", "\t", ";", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else " "


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _split(line: str, delim: str) -> list[str]:
    if delim == " ":
        return line.split()
    return [t.strip() for t in line.split(delim)]


def _find_col(header: Sequence[str], names: Sequence[str]) -> Optional[int]:
    low = [h.strip().lower() for h in header]
    for i, h in enumerate(low):
        if h in names:
            return i
    return None


def load_points(
    path: str,
    *,
    delimiter: Optional[str] = None,
    x_col: Optional[int] = None,
    y_col: Optional[int] = None,
    z_col: Optional[int] = None,
    intensity_col: Optional[int] = None,
    max_points: Optional[int] = None,
) -> PointCloud:
    """CSV/テキストの ToF データを読み込む。

    引数の列番号(0始まり)を指定すればそれを優先。未指定ならヘッダ名、
    それも無ければ最初の数値3列を x, y, z とみなす。
    max_points を指定するとランダムサンプリングして点数を間引く。
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"ファイルが空です: {path}")

    raw_lines = [ln for ln in text.splitlines() if ln.strip() and
                 not ln.lstrip().startswith("#")]
    if not raw_lines:
        raise ValueError(f"有効なデータ行がありません: {path}")

    if delimiter is None:
        delimiter = _sniff_delimiter("\n".join(raw_lines[:20]))

    # ヘッダ判定: 先頭行に数値でないトークンが含まれていればヘッダとみなす
    first = _split(raw_lines[0], delimiter)
    has_header = any(not _is_number(t) for t in first) and len(first) >= 2

    header: list[str] = first if has_header else []
    data_lines = raw_lines[1:] if has_header else raw_lines

    # 列インデックスの決定
    if x_col is None and header:
        x_col = _find_col(header, _X_NAMES)
        y_col = _find_col(header, _Y_NAMES) if y_col is None else y_col
        z_col = _find_col(header, _Z_NAMES) if z_col is None else z_col
        if intensity_col is None:
            intensity_col = _find_col(header, _I_NAMES)

    if x_col is None or y_col is None or z_col is None:
        # 最初の数値3列を採用
        sample = _split(data_lines[0], delimiter)
        numeric_idx = [i for i, t in enumerate(sample) if _is_number(t)]
        if len(numeric_idx) < 3:
            raise ValueError(
                f"数値列が3列未満です(検出 {len(numeric_idx)} 列)。"
                "delimiter や列指定を見直してください。"
            )
        x_col, y_col, z_col = numeric_idx[0], numeric_idx[1], numeric_idx[2]
        if intensity_col is None and len(numeric_idx) >= 4:
            intensity_col = numeric_idx[3]

    cols = [x_col, y_col, z_col]
    if intensity_col is not None:
        cols.append(intensity_col)

    # データ本体を numpy で読み込む(高速・堅牢)
    sep = "," if delimiter == "," else None if delimiter == " " else delimiter
    buf = io.StringIO("\n".join(data_lines))
    try:
        arr = np.loadtxt(buf, delimiter=sep, usecols=cols, ndmin=2)
    except ValueError:
        # 行ごとに不揃いな場合は手動パース
        rows = []
        for ln in data_lines:
            toks = _split(ln, delimiter)
            try:
                rows.append([float(toks[c]) for c in cols])
            except (IndexError, ValueError):
                continue
        if not rows:
            raise ValueError("数値データを解析できませんでした。")
        arr = np.asarray(rows, dtype=float)

    xyz = arr[:, :3]
    intensity = arr[:, 3] if arr.shape[1] >= 4 else None

    # NaN/Inf を除去
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if intensity is not None:
        intensity = intensity[finite]

    if len(xyz) == 0:
        raise ValueError("有効な点が0件でした。")

    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        if intensity is not None:
            intensity = intensity[idx]

    return PointCloud(xyz=xyz, intensity=intensity, source=path)
