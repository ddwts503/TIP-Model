#!/bin/bash
# ダブルクリックで「3D ビューア」を起動するランチャー(macOS)。
# どのSSDのデータでも、起動後にブラウザの画面で選んで読み込めます。
# 終了するには、このウインドウで Control+C を押す。

cd "$HOME/TIP-Model" 2>/dev/null || { echo "TIP-Model フォルダが見つかりません"; read -p "Enterで閉じる"; exit 1; }

echo "==== 3D ビューア ===="
echo "[1/3] 道具を最新版に更新中..."
git fetch origin claude/lucid-cerf-3muz44 2>/dev/null
git reset --hard origin/claude/lucid-cerf-3muz44 2>/dev/null
git checkout claude/lucid-cerf-3muz44 2>/dev/null

echo "[2/3] SSDの計測データ(.dat)を探しています..."
# /Volumes 以下の大きい .dat を新しい順に探す(最初の1つを初期表示。画面で変更可)
DAT=$(find /Volumes -maxdepth 6 -name "*.dat" -size +50M 2>/dev/null \
      | xargs -I{} stat -f "%m %N" {} 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$DAT" ]; then
  echo "計測データ(.dat)が見つかりませんでした。"
  echo "SSDが接続されているか確認してください。"
  read -p "Enterで閉じる"
  exit 1
fi

echo "最初に表示するデータ: $DAT"
echo "(別のSSDのデータは、開いた画面の「データの選択」で選べます)"
echo "[3/3] ブラウザで3Dビューアを開きます。少し待ってください..."
echo "(終了するときは、このウインドウで Control+C)"
python3 -m tof_viz --mode view3d "$DAT"

read -p "終了しました。Enterで閉じる"
