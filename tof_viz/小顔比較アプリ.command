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

echo "[3/4] SSDの計測データ(.dat)を探しています...(数秒)"
# 速い検索: 浅い階層・大文字小文字を無視・最初に見つかった大きい.datを使う
DAT=$(find /Volumes -maxdepth 5 -iname "*.dat" -size +20M 2>/dev/null | head -1)
if [ -z "$DAT" ]; then
  echo "計測データ(.dat)が見つかりませんでした。"
  echo "SSDは認識されていますが、その中に .dat が見当たりません。"
  echo "  接続中のSSD(/Volumes):"
  ls -1 /Volumes 2>/dev/null | sed 's/^/    /'
  echo "  上のSSDの中に .dat ファイルがあるか確認してください。"
  echo "  (このログ「小顔アプリ_ログ.txt」を送ってもらえれば調べます)"
  read -p "Enterで閉じる"
  exit 1
fi
echo "  最初に使うデータ: $DAT"

echo "[4/4] ブラウザで比較アプリを開きます。少しお待ちください…"
echo "      終了するときは、このウインドウで Control+C"
echo "---------------------------------------------"
# アプリが実際の住所を書き出したら、その住所でブラウザを自動で開く
URLFILE="$HOME/.tofviz_url.txt"
rm -f "$URLFILE"
export TOFVIZ_NO_AUTOOPEN=1
(
  for i in $(seq 1 120); do
    if [ -s "$URLFILE" ]; then
      sleep 2
      U=$(cat "$URLFILE")
      echo ">>> ブラウザを開きます: $U"
      open "$U"
      break
    fi
    sleep 1
  done
) &
python3 -m tof_viz --mode compare --after same "$DAT"
echo "---------------------------------------------"
echo "アプリが終了しました。問題があれば、デスクトップの「小顔アプリ_ログ.txt」を送ってください。"
read -p "Enterで閉じる"
