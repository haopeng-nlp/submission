#!/usr/bin/bash
# This script is used to export an optimized model to ONNX format
# Prerequisites: Install Optimum CLI with `pip install "optimum[onnxruntime]"`
# Usage: ./onnx_exporter.sh

# Name of model to be exported on Huggingface ModelHub
OUTPUT_DIR="onnx_models";
HF_MODEL_NAME="facebook/mbart-large-50-many-to-one-mmt"

optimum-cli export onnx \
    --model $HF_MODEL_NAME \
    --task text2text-generation-with-past \
    --framework pt \
    --device cuda \
    --optimize O4 \
    $OUTPUT_DIR/$HF_MODEL_NAME;