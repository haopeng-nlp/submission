import json
import logging
import sys
import argparse

import more_itertools
import torch
import transformers
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBartTokenizer, pipeline
from typing import List


ONNX_PATH='mbart_onnx/'
HF_PATH='facebook/mbart-large-en-ro'
# Submission
class MBART():
    def __init__(
            self,
            pretrained_model_name_or_path='facebook/mbart-large-en-ro',
            use_onnx=False):
        self.use_onnx = use_onnx
        self.prepare(pretrained_model_name_or_path)

    def prepare(self, pretrained_model_name_or_path):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = MBartTokenizer.from_pretrained(HF_PATH, src_lang="en_XX")
        if self.use_onnx:
            self.onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                ONNX_PATH,
                use_io_binding=torch.cuda.is_available()).to(self.device)
            self.onnx_pipeline = pipeline(
                "text2text-generation",
                model=self.onnx_model,
                tokenizer=self.tokenizer,
                max_length=10,
                device=self.device)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(HF_PATH).half().to(self.device)

    def predict(self, inputs: List[str]):
        if self.use_onnx:
            outputs = self.onnx_pipeline(inputs)
            for output in outputs:
                yield output["generated_text"]
        else:
            inputs = self.tokenizer.batch_encode_plus(
                inputs,
                padding=True,
                return_tensors="pt",
                truncation="only_first",
                pad_to_multiple_of=8,
            ).input_ids
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(inputs, max_length=10)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                yield output.strip()

    def predict_offline(self, inputs: List[str]):
        batches = more_itertools.chunked(inputs, 32)
        # inputs = [self._convert_fn(i) for i in inputs]
        for batch in batches:
            inputs = self.tokenizer.batch_encode_plus(
                batch,
                padding=True,
                return_tensors="pt",
                truncation="only_first",
                pad_to_multiple_of=8,
            ).input_ids
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(inputs, max_length=10)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                yield output.strip()
