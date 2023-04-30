import json
import sys
import argparse

import torch
import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer


# Submission
class MBART():
    def __init__(
            self,
            pretrained_model_name_or_path='facebook/mbart-large-en-ro',
            use_onnx=False):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = MBartTokenizer.from_pretrained(
            pretrained_model_name_or_path, src_lang="en_XX")

        self.use_onnx = use_onnx
        if self.use_onnx:
            # TODO: Setup ORT Quantizer and Optimizer
            self.onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path).to(device)
            self.onnx_pipeline = pipeline("text2text-generation", model=self.onnx_model, tokenizer=tknr)
        else
            self.model = MBartForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path).to(device)

        # TODO
        #self._convert_fn = lambda text: text["input"]

    def predict(self, inputs):
        # inputs = [self._convert_fn(i) for i in inputs]
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).input_ids
        inputs = inputs.to(self.model.device)
        if self.use_onnx:
            outputs = pipeline(inputs)
            for output in outputs:
                yield output["generated_text"]
        else:
            outputs = self.model.generate(inputs, max_length=10)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                yield output.strip()
