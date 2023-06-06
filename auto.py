from typing import List
import more_itertools
import torch
from torch import autocast

from transformers import AutoModel, AutoTokenizer

device_type = "cuda" if torch.cuda.is_available() else "cpu"


class AutoSeq2SeqModelSubmission(object):
    """
    Class for Generic HuggingFace Seq2Seq Model for Submission to FairPlay.

    This is intended as an example base class. The example implementations provided
    outline how to use mBART or OPUS Translation models. The only functionality
    we outline here is (A) how to retrieve and set up a tokenizer and model using
    the Hugginface Hub and (B) how the `predict()` and `predict_offline()` functions
    work.

    If your model is a generic HF Seq2Seq / MT model, then this script can work out of
    the box for you. If your model is distinctly different, requires special processing,
    or has some issues with pulling from HF hub: you can subclass this class and extend
    the functionality we provide. You *should* generally only need to modify the `prepare()`
    function for your use case.

    Valid tasks are:
        * wmt16-en-ro
        * wmt16-ro-en
        * wmt14-en-de
        * wmt14-de-en
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        task: str,
        quantize_mode: str,
        offline_bsz: int = 32,
    ) -> None:
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._task = task
        self._quantize_mode = quantize_mode
        self._offline_bsz = offline_bsz

        # In some instances, you might need to add extra arguments to the `model.generate()`
        # functionality below. You can add this here to be injected during inference.
        # See `mbart.py` for an example of this.
        self.additional_args = {}
        self.prepare()

    def prepare(self) -> None:
        # We set up the device to move to GPU during submission.
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Heuristically infer the src and tgt langs from the specified task
        self.src_lang, self.tgt_lang = (
            self._task.split("-")[1],
            self._task.split("-")[2],
        )

        # Declare the tokenizer using AutoTokenizer. SRC and TGT lang are passed as well but
        # may be ignored by the Tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._pretrained_model_name_or_path,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )

        # We assume AutoModel will give you the correct model output.
        # You can change this (e.g., for `AutoModelForSeq2SeqLM`) if required.
        model_cls = AutoModel

        if self._quantize_mode == "bb8" or self._quantize_mode == "bb4":
            raise NotImplementedError(
                "Not all models support `bitsandbytes` quantization so this is not "
                "offered as a default allowable configuration. "
                "See the `mbart.py` script for an example of how to implement this."
            )

        # We provide initial support for FP32, FP16 and BF16 data types
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

        # Set the model to eval mode.
        self.model.eval()

    # If your model does not support `.generate()` then you must
    # subclass and override `predict()` for your use case.

    def predict(self, inputs: List[str]) -> str:
        with torch.autocast(device_type=device_type):
            # Run batch tokenization on a list of string inputs provided by stdout using the benchmark
            inputs = self.tokenizer.batch_encode_plus(
                inputs,
                padding=True,
                return_tensors="pt",
            ).input_ids

            # Move inputs to GPU
            inputs = inputs.to(self.model.device)

            # Generate output predictions.
            outputs = self.model.generate(inputs, **self.additional_args)

            # Decode from IDs to Tokens
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Yield outputs back to stdout
            for output in outputs:
                yield output.strip()

    # Offline prediction is the second usage scenario
    def predict_offline(self, inputs: List[str]) -> str:
        with torch.autocast(device_type=device_type):
            # Processing speed for the benchmark is significantly improved by sorting the inputs
            inputs = sorted(inputs, key=len)

            # Chunk your inputs and heuristically batch to size 32
            # (this will only affect the batch size for the offline setting)
            # Below processing mimics `predict()` above.
            batches = more_itertools.chunked(inputs, self._offline_bsz)

            for batch in batches:
                inputs = self.tokenizer.batch_encode_plus(
                    batch,
                    padding=True,
                    return_tensors="pt",
                ).input_ids
                inputs = inputs.to(self.model.device)
                outputs = self.model.generate(inputs, **self.additional_args)
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for output in outputs:
                    yield output.strip()
