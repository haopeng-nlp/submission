import os
import more_itertools
import torch
from huggingface_hub import hf_hub_download
from typing import List, Tuple
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline
from transformers import (
    MBartTokenizer,
    MBart50Tokenizer,
    M2M100Tokenizer,
    MBartForConditionalGeneration,
    M2M100ForConditionalGeneration,
)
from auto import AutoSeq2SeqModelSubmission

MBART_PIPELINE = (
    MBartTokenizer,
    MBartForConditionalGeneration,
    "decoder_start_token_id",
)
MBART50_PIPELINE = (
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    "forced_bos_token_id",
)
M2M100_PIPELINE = (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    "forced_bos_token_id",
)


# ISO is 5 char "en_XX"
MBART_MODELS = ["facebook/mbart-large-en-ro"]

# ISO is 5 char "en_XX"
MBART50_MODELS = [
    "facebook/mbart-large-50-one-to-many-mmt",  # Useful for EN->X
    "facebook/mbart-large-50-many-to-one-mmt",  # X->EN
    "facebook/mbart-large-50-many-to-many-mmt",  # Any to Any
]

# ISO is 2 char
M2M100_MODELS = [
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "facebook/m2m100-12B-avg-5-ckpt",
    "facebook/m2m100-12B-avg-10-ckpt",
    "facebook/m2m100-12B-last-ckpt",
    "facebook/wmt21-dense-24-wide-en-x",
    "facebook/wmt21-dense-24-wide-x-en",
]

ONNX_MODEL_DIR = "jaredfern/efficiency-benchmark-onnx"
VALID_MODELS = MBART_MODELS + MBART50_MODELS + M2M100_MODELS
PATH2PIPELINE = {m: MBART_PIPELINE for m in MBART_MODELS}
PATH2PIPELINE = {**PATH2PIPELINE, **{m: MBART50_PIPELINE for m in MBART50_MODELS}}
PATH2PIPELINE = {**PATH2PIPELINE, **{m: M2M100_PIPELINE for m in M2M100_MODELS}}

TASK2SRCTGT = {
    "wmt16-en-ro": {"src_lang": "en_XX", "tgt_lang": "ro_RO"},
    "wmt16-ro-en": {"src_lang": "ro_RO", "tgt_lang": "en_XX"},
    "wmt14-en-de": {"src_lang": "en_XX", "tgt_lang": "de_DE"},
    "wmt14-de-en": {"src_lang": "de_DE", "tgt_lang": "en_XX"},
}


def model_task_to_src_tgt_lang(model: str, task: str) -> Tuple[str]:
    """
    Input a model name and the relevant task and returns the source and target language IDs.
    Use TASK2SRCTGT to get the 5 char codes. Trim to 2 char if the model is in M2M100_MODELS.
    Also raises an exception if the combination of model and src_lang, tgt_lang is not valid
    (i.e., the model is not designed for this translation pair)
    """
    src_lang = TASK2SRCTGT.get(task)["src_lang"]
    tgt_lang = TASK2SRCTGT.get(task)["tgt_lang"]

    if model in MBART_MODELS:
        assert (
            src_lang == "en_XX"
        ), f"Model {model} does not support src_lang: {src_lang}"
        assert (
            tgt_lang == "ro_RO"
        ), f"Model {model} does not support tgt_lang: {tgt_lang}"

    if (model == MBART50_MODELS[0]) or (model == M2M100_MODELS[5]):  # One to many
        assert (
            src_lang == "en_XX"
        ), f"Model {model} does not support src_lang: {src_lang}"

    if (model == MBART50_MODELS[1]) or (model == M2M100_MODELS[6]):  # Many to one
        assert (
            tgt_lang == "en_XX"
        ), f"Model {model} does not support tgt_lang: {tgt_lang}"

    if model in M2M100_MODELS:
        src_lang = src_lang[:2]
        tgt_lang = tgt_lang[:2]

    return src_lang, tgt_lang


# Submission
class MBART(AutoSeq2SeqModelSubmission):
    """
    A wrapper for HuggingFace mBART models for inference prediction. We currently support only those in VALID_MODELS.
    We allow for quantization using either torch_dtype=torch.float16 or the `bits_and_bytes` load_in_4bit, load_in_8bit calls.
    For this to function correctly, you need bleeding edge Transformers and Accelerate. Pass this to `efficiency-benchmark`
    using a `requirements.txt` file with the argument `--pip`. We provide a reasonable initial configuration for runtime
    on 1x RTX A6000 GPU.

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
        use_onnx: bool,
    ) -> None:
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._task = task
        self._quantize_mode = quantize_mode
        self._use_onnx = use_onnx
        self.prepare()

    def prepare(self) -> None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.src_lang, self.tgt_lang = model_task_to_src_tgt_lang(
            model=self._pretrained_model_name_or_path, task=self._task
        )

        tokenizer_cls, model_cls, add_arg_key = PATH2PIPELINE[
            self._pretrained_model_name_or_path
        ]

        self.tokenizer = tokenizer_cls.from_pretrained(
            self._pretrained_model_name_or_path,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )

        self.tokenizer.src_lang = self.src_lang  # Setup source language

        if self._pretrained_model_name_or_path in M2M100_MODELS:
            self.src_lang_id = self.tokenizer.get_lang_id(self.src_lang)
            self.tgt_lang_id = self.tokenizer.get_lang_id(self.tgt_lang)
        else:
            self.src_lang_id = self.tokenizer.lang_code_to_id[self.src_lang]
            self.tgt_lang_id = self.tokenizer.lang_code_to_id[self.tgt_lang]

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
            try:
                # Manual download because `.from_pretrained()` fails to download of encoder data file
                hf_hub_download(repo_id=ONNX_MODEL_DIR, filename=f"{self._pretrained_model_name_or_path}/encoder_model.onnx.data")
                hf_hub_download(repo_id=ONNX_MODEL_DIR, filename=f"{self._pretrained_model_name_or_path}/decoder_model.onnx.data")
            except:
                pass

            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                ONNX_MODEL_DIR,
                subfolder=self._pretrained_model_name_or_path,
                provider="CUDAExecutionProvider",
                use_io_binding=torch.cuda.is_available(),
                generation_config=self.additional_args,
                use_merged=True,
                use_cache=True
            ).to(device)
        else:
            if self._quantize_mode == "fp16":
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path, torch_dtype=torch.float16
                ).to(device)
            elif self._quantize_mode == "bf16":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path, torch_dtype=torch.bfloat16
                ).to(device)
            elif self._quantize_mode == "bb8":
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path,
                    device_map="auto",
                    load_in_8bit=True,
                )
            elif self._quantize_mode == "bb4":
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path,
                    device_map="auto",
                    load_in_4bit=True,
                )
            else:
                self.model = model_cls.from_pretrained(
                    self._pretrained_model_name_or_path
                ).to(device)

            self.model.eval()

    def predict(self, inputs: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs, **self.additional_args)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output in outputs:
            yield output.strip()

    def predict_offline(self, inputs: List[str]):
        inputs = sorted(inputs, key=len)
        batches = more_itertools.chunked(inputs, 32)
        for batch in batches:
            inputs = self.tokenizer.batch_encode_plus(
                batch,
                padding=True,
                return_tensors="pt",
                truncation="only_first",
                pad_to_multiple_of=8,
            ).input_ids
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(inputs, **self.additional_args)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                yield output.strip()

