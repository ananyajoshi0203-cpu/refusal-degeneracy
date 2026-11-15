#!/bin/bash
# Runs locally: start (or reuse) an A100 session, mount Drive so results outlive the VM, ship
# the committed repo plus the run's splits, and start run.sh.
# Usage: launch.sh <config> <local splits.json> <session> <stage>...   (stages: see run.sh)
set -euo pipefail
config=$1
splits=$2
session=$3
shift 3
run=$(basename "$config" .json)
root=/content/drive/MyDrive/refusal-degeneracy/results
results=$root/$run
bundle=$(mktemp -d)

git archive --format=tar HEAD > "$bundle/repo.tar"
tar -rf "$bundle/repo.tar" -C "$(dirname "$splits")" "$(basename "$splits")"
gzip "$bundle/repo.tar"

colab sessions 2>/dev/null | grep -q "\b$session\b" || colab new -s "$session" --gpu A100 --high-mem
colab ls -s "$session" /content/drive/MyDrive >/dev/null 2>&1 || colab drivemount -s "$session"
colab upload -s "$session" "$bundle/repo.tar.gz" /content/repo.tar.gz
colab upload -s "$session" "$(dirname "$0")/setup.sh" /content/setup.sh
colab upload -s "$session" "$(dirname "$0")/run.sh" /content/run.sh
echo "!bash /content/setup.sh $results" | colab exec -s "$session"
echo "!bash /content/run.sh $config $root $*" | colab exec -s "$session"
echo "results: $results"
