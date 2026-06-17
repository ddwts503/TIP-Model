#!/bin/bash
# ダブルクリックで「ビフォー・アフター 小顔チェック」を起動するランチャー(macOS)。
# SSD内の .dat を自動で探し、ブラウザで比較アプリを開く。
# 終了するには、このウインドウで Control+C を押す。

cd "$HOME/TIP-Model" 2>/dev/null || { echo "TIP-Model フォルダが見つかりません"; read -p "Enterで閉じる"; exit 1; }

echo "==== 小顔比較アプリ ===="
echo "[1/3] 道具を最新版に更新中..."
git pull origin claude/lucid-cerf-3muz44 2>/dev/null

echo "[2/3] SSDの計測データ(.dat)を探しています..."
# /Volumes 以下の大きい .dat を新しい順に探す
DAT=$(find /Volumes -maxdepth 6 -name "*.dat" -size +50M 2>/dev/null \
      | xargs -I{} stat -f "%m %N" {} 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$DAT" ]; then
  echo "計測データ(.dat)が見つかりませんでした。"
  echo "SSDが接続されているか確認してください。"
  read -p "Enterで閉じる"
  exit 1
fi

echo "使うデータ: $DAT"
echo "[3/3] ブラウザで比較アプリを開きます。少し待ってください..."
echo "(終了するときは、このウインドウで Control+C)"
python3 -m tof_viz --mode compare --after same "$DAT"

read -p "終了しました。Enterで閉じる"
