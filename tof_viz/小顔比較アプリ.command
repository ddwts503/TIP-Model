#!/bin/bash
# ダブルクリックで「ビフォー・アフター 小顔チェック」を起動するランチャー(macOS)。
# すべての出力をデスクトップの「小顔アプリ_ログ.txt」に記録します(不具合時に送ってください)。
# 終了するには、このウインドウで Control+C を押す。

LOG="$HOME/Desktop/小顔アプリ_ログ.txt"
exec > >(tee "$LOG") 2>&1   # 画面表示とログ記録を両方行う

echo "==== 小顔比較アプリ ===="
echo "日時: $(date)"

cd "$HOME/TIP-Model" 2>/dev/null || { echo "TIP-Model フォルダが見つかりません"; read -p "Enterで閉じる"; exit 1; }

echo "[1/4] 最新版に更新中..."
git fetch origin claude/lucid-cerf-3muz44 2>&1
git reset --hard origin/claude/lucid-cerf-3muz44 2>&1
git checkout claude/lucid-cerf-3muz44 2>&1
echo "  現在のバージョン: $(git log -1 --oneline)"

echo "[2/4] 古いアプリを停止..."
pkill -f "tof_viz" 2>/dev/null
sleep 1

echo "[3/4] SSDの計測データ(.dat)を探しています..."
DAT=$(find /Volumes -maxdepth 6 -name "*.dat" -size +50M 2>/dev/null \
      | xargs -I{} stat -f "%m %N" {} 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$DAT" ]; then
  echo "計測データ(.dat)が見つかりませんでした。SSDの接続を確認してください。"
  read -p "Enterで閉じる"
  exit 1
fi
echo "  最初に使うデータ: $DAT"

echo "[4/4] ブラウザで比較アプリを開きます。下に出る http://… が画面の住所です。"
echo "      (自動で開かない時は、その住所をChromeのアドレス欄に入れてください)"
echo "      終了するときは、このウインドウで Control+C"
echo "---------------------------------------------"
python3 -m tof_viz --mode compare --after same "$DAT"
echo "---------------------------------------------"
echo "アプリが終了しました。問題があれば、デスクトップの「小顔アプリ_ログ.txt」を送ってください。"
read -p "Enterで閉じる"
