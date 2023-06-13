import argparse
import json
import sys
from functools import partial
from typing import List, Tuple

import more_itertools
import torch
import transformers
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, pipeline
from auto import AutoSeq2SeqModelSubmission

ONNX_MODEL_DIR = "jaredfern/efficiency-benchmark-onnx"

class OPUS(AutoSeq2SeqModelSubmission):
    """
    Class for OPUS Translation for Efficiency Benchmark.
    Inherits from MBART class (which does most of the processing).
    The main difference is how model + tokenizer is loaded.
    """

    def __init__(
        self, pretrained_model_name_or_path: str, task: str, quantize_mode: str,
        use_onnx: bool
    ):
        self._use_onnx = use_onnx
        super().__init__(pretrained_model_name_or_path, task, quantize_mode)

    def prepare(self) -> None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.src_lang, self.tgt_lang = (
            self._task.split("-")[1],
            self._task.split("-")[2],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._pretrained_model_name_or_path,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )


        model_cls = MarianMTModel

        self.additional_args = {
            "max_new_tokens": 200,
            "early_stopping": False,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "temperature": 1.0,
            add_arg_key: self.tgt_lang_id
        }
        if self._use_onnx:
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                ONNX_MODEL_DIR,
                subfolder=self._pretrained_model_name_or_path,
                provider="CUDAExecutionProvider",
                use_io_binding=torch.cuda.is_available(),
                use_merged=True,
                use_cache=True
            ).to(device)
        else:
            model_cls = MarianMTModel

            if self._quantize_mode == "bb8" or self._quantize_mode == "bb4":
                raise ValueError("Lower quantization is not supported for OPUS models.")

            if self._quantize_mode == "fp16":
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path, torch_dtype=torch.float16
                ).to(device)
            elif self._quantize_mode == "bf16":
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path, torch_dtype=torch.bfloat16
                ).to(device)
            else:
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path
                ).to(device)

            self.model.eval()
