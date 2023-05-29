import more_itertools
import torch
from typing import List
from transformers import MBartForConditionalGeneration, MBartTokenizer

TASK2SRCTGT = {"wmt16-en-ro": {"src_lang": "en_XX", "tgt_lang": "ro_RO"},
               "wmt16-en-de": {"src_lang": "en_XX", "tgt_lang": "de_DE"},
               "wmt16-ro-en": {"src_lang": "ro_RO", "tgt_lang": "en_XX"},
               "wmt16-de-en": {"src_lang": "de_DE", "tgt_lang": "en_XX"}
               }

# Submission
class MBART():
    def __init__(self, task, quantize_mode):
        src_tgt = TASK2SRCTGT.get(task)
        self.src_lang = "en_XX"#src_tgt['src_lang']
        self.tgt_lang = "ro_RO"#src_tgt['tgt_lang']
        self.pretrained_model_name_or_path = "facebook/mbart-large-en-ro"
        self._quantize_mode = quantize_mode
        self._use_bb8 = self._quantize_mode == "bb8"
        self._use_bb4 = self._quantize_mode == "bb4"
        self._use_fp16 = self._quantize_mode == "half"
        self.prepare()

    def prepare(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", 
            src_lang=self.src_lang, 
            tgt_lang=self.tgt_lang)

        if self._use_fp16:
            print("Declaring half precision model")
            self.model = MBartForConditionalGeneration.from_pretrained(self.pretrained_model_name_or_path).half().to(device)
        elif self._use_bb8:
            print("Declaring 8-bit precision model")
            self.model = MBartForConditionalGeneration.from_pretrained(self.pretrained_model_name_or_path, device_map="auto", load_in_8bit=True)
        elif self._use_bb4:
            print("Declaring 4-bit precision model")
            self.model = MBartForConditionalGeneration.from_pretrained(self.pretrained_model_name_or_path, device_map="auto", load_in_4bit=True)
        else:
            print("No model weight quantization selected. Loading in full-precision")
            self.model = MBartForConditionalGeneration.from_pretrained(self.pretrained_model_name_or_path).to(device)

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
