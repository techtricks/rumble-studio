#!/bin/zsh
# ytauto launcher for launchd — one full cycle + Rumble upload, hourly.
# launchd expects absolute paths; it starts us with no shell environment.

cd "/Users/mac/Documents/rumble" || exit 1

LOG="data/logs/launchd.log"
LOCK="data/lock"
mkdir -p data/logs

# --- single-instance guard: a previous cycle may still be rendering ---
if [[ -f "$LOCK" ]]; then
    OLDPID=$(cat "$LOCK" 2>/dev/null)
    if [[ -n "$OLDPID" ]] && kill -0 "$OLDPID" 2>/dev/null; then
        echo "=== cycle skipped $(date '+%F %T'): pid $OLDPID still running ===" >> "$LOG"
        exit 0
    fi
    rm -f "$LOCK"   # stale lock from a crashed cycle
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

{
    echo ""
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') cycle start ==="
} >> "$LOG"

# 1. discover → script → approve (AUTO_APPROVE=1) → render → upload stage
.venv/bin/python -m ytauto run --cycles 1 >> "$LOG" 2>&1

# 2. push anything newly rendered to Rumble (no-op when nothing is rendered)
.venv/bin/python -m ytauto rumble >> "$LOG" 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') cycle end ===" >> "$LOG"
