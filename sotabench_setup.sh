#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
$PYTHON -m pip install torch
$PYTHON -m pip install torchvision
$PYTHON -m pip install scipy
