#!/usr/bin/env bash

mkdir original_tf
cd original_tf
touch __init__.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/efficientnet_builder.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/efficientnet_model.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/eval_ckpt_main.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/utils.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/preprocessing.py
cd ..
mkdir -p tmp