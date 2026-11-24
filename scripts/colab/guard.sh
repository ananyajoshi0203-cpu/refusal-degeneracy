#!/bin/bash
# Runs locally beside a chain started by launch.sh. Every <seconds> it copies the run's small
# result files (JSON, CSV, reply JSONL, run.log) off the VM, and once the Colab balance drops
# below <reserve> units it copies them a last time and stops the session: when the units run
# out Colab deletes the VM, so without this every result not yet copied is lost. It also stops
# the session when the chain writes exit_code, so a finished run does not keep billing; set
# KEEP=1 to leave it running.
# Usage: guard.sh <session> <VM results root> <local dir> [reserve units] [seconds]
set -uo pipefail
session=$1
remote=$2
dest=$3
reserve=${4:-10}
seconds=${5:-600}

pull() {
  printf '%s\n' "!find $remote -type f \( -name '*.json' -o -name '*.csv' -o -name '*.jsonl' -o -name run.log -o -name exit_code \)" |
    colab exec -s "$session" --timeout 90 2>/dev/null | grep "^$remote/" | while read -r path; do
      mkdir -p "$dest/$(dirname "${path#"$remote"/}")"
      colab download -s "$session" "$path" "$dest/${path#"$remote"/}" >/dev/null 2>&1
    done
}

balance() {
  colab usage 2>/dev/null | awk '/balance/ {print $3}'
}

stop() {
  pull
  [ "${KEEP:-0}" = 1 ] || colab stop -s "$session"
}

while true; do
  pull
  units=$(balance)
  echo "$(date '+%F %T') balance ${units:-unknown}; $(tail -c 120 "$dest/run.log" 2>/dev/null | tr '\r\n' '  ' | tail -c 100)"
  if [ -f "$dest/exit_code" ]; then
    echo "chain finished with exit code $(cat "$dest/exit_code")"
    stop
    exit 0
  fi
  if [ -n "$units" ] && awk -v u="$units" -v r="$reserve" 'BEGIN {exit !(u < r)}'; then
    echo "balance $units is below the $reserve-unit reserve; stopping before Colab deletes the VM"
    stop
    exit 1
  fi
  sleep "$seconds"
done
