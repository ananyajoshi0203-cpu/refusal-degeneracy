#!/bin/bash
# Runs on the Colab VM: unpack the repo bundle and install it. Usage: setup.sh <results dir>
set -euo pipefail
results=$1
cd /content
rm -rf refusal-degeneracy && mkdir refusal-degeneracy && tar -xzf repo.tar.gz -C refusal-degeneracy
cd refusal-degeneracy
pip install -q -e . 2>&1 | tail -2
mkdir -p "$results"
[ -f "$results/splits.json" ] || cp splits.json "$results/splits.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
