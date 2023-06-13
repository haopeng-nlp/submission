#!/usr/bin/bash
# This script is used to export an optimized model to ONNX format
# Prerequisites: Install Optimum CLI with `pip install "optimum[onnxruntime]"`
# Usage: ./onnx_exporter.sh

# Name of model to be exported on Huggingface ModelHub
OUTPUT_DIR="/data/jaredfer/efficiency-benchmark-onnx"

for HF_MODEL_NAME in \
    "facebook/wmt21-dense-24-wide-x-en" \
    "facebook/m2m100-12B-avg-5-ckpt" \
    "facebook/mbart-large-50-many-to-many-mmt" \
    "facebook/mbart-large-50-many-to-one-mmt" \
    "Helsinki-NLP/opus-mt-de-en" \
    "facebook/m2m100_1.2B" \
    "facebook/m2m100_418M"
do
    echo "EXPORTING $HF_MODEL_NAME"
    optimum-cli export onnx \
        --model $HF_MODEL_NAME \
        --task text2text-generation-with-past \
        --framework pt \
        --optimize O3 \
        $OUTPUT_DIR/$HF_MODEL_NAME;
done;
