#!/bin/bash
# Runs on the Colab VM, detached. Usage: run.sh <config> <results root> <stage>...
# Stages, run in order: base (weeks 1-2, geometry and cluster ablation on the undefended model), dsa, neuronguard
# (train the defense, then weeks 1-2 + geometry on it). Results land in <root>/<run>[-<defense>]
# on Drive; defended weights stay on the VM disk. Progress in <root>/run.log, exit code in
# <root>/exit_code once every stage has finished. REPO, WORK and DEFENSE_ARGS (extra flags for
# the defense scripts) can be overridden for a smoke run off Colab.
set -uo pipefail
config=$1
root=$2
shift 2
run=$(basename "$config" .json)
model=$(python -c "import json; print(json.load(open('$config'))['model'])")
base=$root/$run
work=${WORK:-/content/work}
defense_args=${DEFENSE_ARGS:-}
cd "${REPO:-/content/refusal-degeneracy}"
rm -f "$root/exit_code"

attack() {  # attack <model> <results dir>
  python scripts/run_week1.py --config "$config" --model "$1" --out "$2" --resume &&
  python scripts/run_geometry.py --run "$2" &&
  python scripts/run_conditional.py --run "$2" --resume
}

defend() {  # defend <defense>
  local out=$root/$run-$1
  mkdir -p "$out" && cp "$base/splits.json" "$out/splits.json"
  python "scripts/run_$1.py" --model "$model" --out "$work/$run-$1" --splits "$base/splits.json" --resume $defense_args &&
  cp "$work/$run-$1"/*.json "$out/" &&
  attack "$work/$run-$1/model" "$out"
}

stages() {
  for stage in "$@"; do
    echo "=== $stage $(date)"
    case $stage in
      base) attack "$model" "$base" ;;
      dsa|neuronguard) defend "$stage" ;;
      *) echo "unknown stage $stage"; return 2 ;;
    esac || return
  done
}

export -f attack defend stages
export config root run model base work defense_args
nohup bash -c "stages $*; echo \$? > $root/exit_code" >> "$root/run.log" 2>&1 &
echo "started pid $!"
