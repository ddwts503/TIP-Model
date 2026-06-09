# tof_viz — 3D ToF データの可視化・輪切りツール

CSV / テキスト形式の ToF(Time-of-Flight)点群データを読み込み、

- **3D 表示**(matplotlib 散布図、または Open3D インタラクティブ表示)
- **輪切り(断面)表示**:任意軸(X / Y / Z)に沿った断面を抽出

する小さな Python ツールです。

## インストール

```bash
pip install -r tof_viz/requirements.txt
```

`numpy` と `matplotlib` のみで動作します。`open3d` はインタラクティブな
3D 表示を使う場合だけ必要です(任意)。

## データ形式

カンマ / タブ / スペース / セミコロン区切りのテキストに対応し、
区切り文字・ヘッダ行・列構成を**自動判定**します。

- ヘッダがある場合: `x, y, z`(および `intensity` / `amplitude` など)の
  列名を自動認識
- ヘッダが無い場合: 最初の数値3列を `x, y, z`、4列目を強度とみなす
- 自動判定がうまくいかない場合は `--delimiter` や `--cols` で明示指定

例(`x,y,z,intensity`):

```csv
x,y,z,intensity
0.512,-0.034,1.230,0.87
...
```

## 使い方(CLI)

```bash
# 3D 点群表示(高さで色付け)
python -m tof_viz data.csv --mode 3d

# 3D を画像保存(GUI 不要 / ヘッドレス環境向け)
python -m tof_viz data.csv --mode 3d --save view3d.png

# Open3D でインタラクティブ 3D 表示
python -m tof_viz data.csv --mode 3d --backend open3d

# 単一の輪切り: Z=1.2 を中心に厚み 0.1 のスラブ
python -m tof_viz data.csv --mode slice --axis z --pos 1.2 --thickness 0.1

# 輪切り一覧: Z 軸方向に 9 枚の断面をグリッド表示
python -m tof_viz data.csv --mode grid --axis z --n 9 --save slices.png

# X 軸方向の断面も可能
python -m tof_viz data.csv --mode grid --axis x --n 6

# スライダーで断面位置を動かすインタラクティブ表示(GUI 必須)
python -m tof_viz data.csv --mode interactive --axis z
```

### 主なオプション

| オプション | 説明 |
|-----------|------|
| `--mode` | `3d` / `slice` / `grid` / `interactive` |
| `--axis` | 輪切りの軸 `x` / `y` / `z`(default: z) |
| `--pos` | `slice` の断面位置(未指定なら中央値) |
| `--thickness` | 断面の厚み(スラブ幅) |
| `--n` | `grid` の断面枚数(default: 9) |
| `--color-by` | `z`(高さ)/ `intensity`(強度)/ `mono` |
| `--backend` | `matplotlib` / `open3d`(3d モード) |
| `--max-points` | 読み込む最大点数(間引き、大規模データ向け) |
| `--delimiter` | 区切り文字(未指定なら自動判定) |
| `--cols` | `x,y,z[,intensity]` の列番号(0始まり)を明示 |
| `--save` | 表示せず画像ファイルに保存 |

## サンプルデータで試す

実データが無くても、擬似 ToF 点群(円柱+球+床)を生成して試せます。

```bash
python tof_viz/examples/generate_sample.py -o sample_tof.csv -n 30000

python -m tof_viz sample_tof.csv --mode 3d --save view3d.png
python -m tof_viz sample_tof.csv --mode grid --axis z --n 9 --save slices.png
```

## Python から使う

```python
from tof_viz import load_points, show_3d, show_slice, show_slices_grid

pc = load_points("data.csv", max_points=200_000)
show_3d(pc, color_by="z")
show_slice(pc, axis="z", position=1.2, thickness=0.1)
show_slices_grid(pc, axis="z", n=12)
```
