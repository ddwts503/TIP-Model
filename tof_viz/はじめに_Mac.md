# はじめての人向け:Mac で 3D ToF データを表示する手順

このページは、プログラミングが初めての方でも、上から順にコピペするだけで
ToF データを 3D 表示・輪切り(断面)表示できるように書いています。
あせらず、1つずつ進めれば大丈夫です。

---

## ステップ1:ターミナルを開く

「ターミナル」という、文字だけの黒い(または白い)画面のアプリを使います。

1. キーボードの **`command(⌘)` を押しながら `space`** を押す
2. 画面中央に出た検索窓に **`ターミナル`** と打って **Enter**
3. 文字だけの画面が開きます ← これが「ターミナル」です

> この先、「貼り付け」は **`command(⌘)` + `V`**、実行は **Enter** です。

---

## ステップ2:準備(最初の1回だけ)

下の枠の中をまるごとコピーして、ターミナルに貼り付けて **Enter**。
ダウンロードや準備が進みます(数分かかることがあります。文字が流れますが見ているだけでOK)。

```bash
cd ~ && git clone https://github.com/ddwts503/TIP-Model.git && cd TIP-Model && git checkout claude/lucid-cerf-3muz44 && pip3 install -r tof_viz/requirements.txt
```

> もし `git` が無いと言われたら、画面の指示に従って「インストール」を押すか、
> [Apple の開発者ツール](https://developer.apple.com/xcode/) の案内に従ってください。
> 2回目以降は、このステップは不要です(下のステップ3だけでOK)。

---

## ステップ3:お試し表示(自分のデータが無くてもOK)

まず、練習用の擬似データを作って表示してみます。下をコピペして **Enter**。

```bash
cd ~/TIP-Model
python3 tof_viz/examples/generate_sample.py -o sample_tof.csv
python3 -m tof_viz sample_tof.csv --mode 3d
```

うまくいくと、**3D のグラフの窓が開いて、マウスでグリグリ回せます**。
窓を閉じれば、また入力できる状態に戻ります。

### 輪切り(断面)も試す

```bash
# Z方向に9枚の断面を並べて表示
python3 -m tof_viz sample_tof.csv --mode grid --axis z --n 9

# スライダーで断面の位置を動かす(一番わかりやすい)
python3 -m tof_viz sample_tof.csv --mode interactive --axis z
```

---

## ステップ4:自分の ToF データ(SSD の中)を表示する

`sample_tof.csv` の部分を、自分のファイルに置き換えるだけです。
**パスの入力はドラッグ&ドロップが簡単**です。

1. ターミナルに、まず次のように途中まで打つ(最後にスペースを1つ入れる):
   ```
   python3 -m tof_viz 
   ```
2. その状態で、**SSD の中の CSV ファイルを、ターミナルの画面にドラッグ&ドロップ**
   → ファイルの場所(パス)が自動で入力されます
3. 続けて表示方法を打って Enter:
   ```
    --mode 3d
   ```

最終的にこんな形になります(パスは自動で入った文字列):

```bash
python3 -m tof_viz "/Volumes/SSDの名前/フォルダ/data.csv" --mode 3d
```

> 外付け SSD は、Mac では `/Volumes/` の下にあります。
> Finder でファイルを右クリック →（`option` キーを押しながら)「パス名をコピー」でも取れます。

---

## 表示の種類(モード)早見表

| やりたいこと | コマンドの `--mode` 部分 | 例 |
|---|---|---|
| 3D で点群を見る | `--mode 3d` | `python3 -m tof_viz データ.csv --mode 3d` |
| 輪切りを並べて見る | `--mode grid` | `python3 -m tof_viz データ.csv --mode grid --axis z --n 9` |
| 1枚だけ輪切り | `--mode slice` | `python3 -m tof_viz データ.csv --mode slice --axis z --pos 1.2 --thickness 0.1` |
| スライダーで断面を動かす | `--mode interactive` | `python3 -m tof_viz データ.csv --mode interactive --axis z` |

- `--axis` は切る向き。`z`(高さ方向)/ `x` / `y` から選べます。
- データがとても大きくて重いときは `--max-points 200000` を足すと軽くなります。
- 窓を開かず画像ファイルに保存したいときは `--save out.png` を足します。

---

## うまくいかないとき

- **赤い文字(エラー)が出て止まった** → その画面をコピーするか写真を撮って残しておく。
- **`python3` が無いと言われた** → ステップ2をやり直すか、Mac に Python を入れる必要があります。
- **データがうまく読めない**(列がずれる等) → ファイルの最初の数行を確認。
  列を手で指定するには `--cols 0,1,2,3`(x,y,z,強度 の列番号、0から数える)を足します。
  区切りがタブの場合は `--delimiter $'\t'` を足します。

詳しい説明は [`tof_viz/README.md`](README.md) にもあります。
