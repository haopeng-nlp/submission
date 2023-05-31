import more_itertools
import torch
from typing import List
from transformers import MBartForConditionalGeneration, MBartTokenizer


# Submission
class MBART():
    def __init__(self):
        self.prepare()

    def prepare(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro").half().to(device)
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")

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
