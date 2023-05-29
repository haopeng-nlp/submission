import more_itertools
import torch
from typing import List
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MBART_PIPELINE = (MBartTokenizer, MBartForConditionalGeneration)
MBART50_PIPELINE = (MBart50Tokenizer, MBartForConditionalGeneration)
M2M100_PIPELINE = (M2M100Tokenizer, M2M100ForConditionalGeneration)

MBART_MODELS = [
    "facebook/mbart-large-en-ro",
    "facebook/mbart-large-cc25" 
]

MBART50_MODELS = [
    "facebook/mbart-large-50-one-to-many-mmt" # Useful for EN->X
    "facebook/mbart-large-50-many-to-one-mmt" # X->EN
    "facebook/mbart-large-50-many-to-many-mmt" # Any to Any
]

M2M100_MODELS = [
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "facebook/m2m100-12B-avg-5-ckpt",
    "facebook/m2m100-12B-avg-10-ckpt",
    "facebook/m2m100-12B-last-ckpt"
]

VALID_MODELS = MBART_MODELS + MBART50_MODELS + M2M100_MODELS

TASK2SRCTGT = {"wmt16-en-ro": {"src_lang": "en_XX", "tgt_lang": "ro_RO"},
               "wmt16-en-de": {"src_lang": "en_XX", "tgt_lang": "de_DE"},
               "wmt16-ro-en": {"src_lang": "ro_RO", "tgt_lang": "en_XX"},
               "wmt16-de-en": {"src_lang": "de_DE", "tgt_lang": "en_XX"}
               }

# Submission
class MBART():
    def __init__(self, pretrained_model_name_or_path: str, task: str, quantize_mode: str) -> None:
        assert pretrained_model_name_or_path in VALID_MODELS, f"Argument {pretrained_model_name_or_path} not recognized!"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.src_lang = TASK2SRCTGT.get(task)['src_lang']
        self.tgt_lang = TASK2SRCTGT.get(task)['tgt_lang']

        self._quantize_mode = quantize_mode
        self._use_bb8 = self._quantize_mode == "bb8"
        self._use_bb4 = self._quantize_mode == "bb4"
        self._use_fp16 = self._quantize_mode == "half"

        if self.pretrained_model_name_or_path in MBART_MODELS:
            tokenizer_cls, model_cls = MBART_PIPELINE
        if self.pretrained_model_name_or_path in MBART50_MODELS:
            tokenizer_cls, model_cls = MBART50_PIPELINE
        if self.pretrained_model_name_or_path in M2M100_MODELS:
            tokenizer_cls, model_cls = M2M100_PIPELINE
        self.prepare(tokenizer_cls, model_cls)

    def prepare(self, 
                tokenizer_cls: transformers.Tokenizer, 
                model_cls: transformers.PreTrainedModel
        ) -> None:

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = tokenizer_cls.from_pretrained(self.pretrained_model_name_or_path, 
            src_lang=self.src_lang, 
            tgt_lang=self.tgt_lang)

        if self._use_fp16:
            print("Declaring half precision model")
            self.model = model_cls.from_pretrained(self.pretrained_model_name_or_path).half().to(device)
        elif self._use_bb8:
            print("Declaring 8-bit precision model")
            self.model = model_cls.from_pretrained(self.pretrained_model_name_or_path, device_map="auto", load_in_8bit=True)
        elif self._use_bb4:
            print("Declaring 4-bit precision model")
            self.model = model_cls.from_pretrained(self.pretrained_model_name_or_path, device_map="auto", load_in_4bit=True)
        else:
            print("No model weight quantization selected. Loading in full-precision")
            self.model = model_cls.from_pretrained(self.pretrained_model_name_or_path).to(device)
            
        self.model.eval()
        
    def predict(self, inputs: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs)
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
            ).input_ids
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(inputs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                yield output.strip()
