#!/usr/bin/env bash
# Free demo of the swap report: replays the saved 300-prompt live run (Qwen3-30B vs Claude
# Sonnet 5.5, judged by Gemini 3.6 Flash) entirely from the local call cache. No network calls.
# Then screenshots the HTML report into docs/img/report.png with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=$(ls -d runs/live-300-* | sort | tail -1)
uv run safeswap export-run "$RUN" --out runs/demo/traces.jsonl
uv run safeswap replay --logs runs/demo/traces.jsonl \
  --candidate qwen/qwen3-30b-a3b-instruct-2507 \
  --judge google/gemini-3.6-flash \
  --baseline-label "claude-sonnet-5.5" \
  --policy examples/policy.yaml \
  --cache-only --budget 0

OUT=$(ls -d runs/replay-* | sort | tail -1)
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --window-size=1120,1500 --screenshot="$PWD/docs/img/report.png" \
    "file://$PWD/$OUT/report.html" 2>/dev/null
  echo "screenshot -> docs/img/report.png"
fi
