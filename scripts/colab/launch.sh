#!/bin/bash
# Runs locally: start (or reuse) an A100 session, ship the committed repo, the run's splits and
# the local Hugging Face token (the Llama models and Llama Guard are gated), and start run.sh.
# Results stay on the VM disk, since mounting Drive from the CLI hangs; run guard.sh beside it
# to copy them off and stop the session before the units run out.
# Usage: launch.sh <config> <local splits.json> <session> <stage>...   (stages: see run.sh)
set -euo pipefail
config=$1
splits=$2
session=$3
shift 3
run=$(basename "$config" .json)
root=/content/results
results=$root/$run
bundle=$(mktemp -d)

git archive --format=tar HEAD > "$bundle/repo.tar"
tar -rf "$bundle/repo.tar" -C "$(dirname "$splits")" "$(basename "$splits")"
gzip "$bundle/repo.tar"

colab sessions 2>/dev/null | grep -q "\b$session\b" || colab new -s "$session" --gpu A100 --high-mem
hf_token=${HF_HOME:-$HOME/.cache/huggingface}/token
if [ -f "$hf_token" ]; then
  echo "!mkdir -p /root/.cache/huggingface" | colab exec -s "$session" --timeout 90
  colab upload -s "$session" "$hf_token" /root/.cache/huggingface/token
fi
colab upload -s "$session" "$bundle/repo.tar.gz" /content/repo.tar.gz
colab upload -s "$session" "$(dirname "$0")/setup.sh" /content/setup.sh
colab upload -s "$session" "$(dirname "$0")/run.sh" /content/run.sh
echo "!bash /content/setup.sh $results" | colab exec -s "$session" --timeout 900
echo "!bash /content/run.sh $config $root $*" | colab exec -s "$session" --timeout 90
echo "results: $results; guard: $(dirname "$0")/guard.sh $session $root <local dir>"
