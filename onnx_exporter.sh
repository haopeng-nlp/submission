#!/usr/bin/bash
# This script is used to export an optimized model to ONNX format
# Prerequisites: Install Optimum CLI with `pip install "optimum[onnxruntime]"`
# Usage: ./onnx_exporter.sh

# Name of model to be exported on Huggingface ModelHub
HF_MODEL_NAME="facebook/mbart-large-en-ro";
OUTPUT_DIR="mbart_onnx";

optimum-cli export onnx \
    --model $HF_MODEL_NAME \
    --task text2text-generation-with-past \
    --framework pt \
    --device cuda \
    --optimize O4 \
    $OUTPUT_DIR;