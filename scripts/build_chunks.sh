#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python tools/build_training_chunks.py --split train --profile talk2dino_metric3d --num-workers 2

PYTHONPATH=. python tools/build_training_chunks.py --split val --profile talk2dino_metric3d --num-workers 2
