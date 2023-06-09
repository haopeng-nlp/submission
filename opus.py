import torch
from transformers import AutoTokenizer, MarianMTModel
from auto import AutoSeq2SeqModelSubmission


class OPUS(AutoSeq2SeqModelSubmission):
    """
    Class for OPUS Translation for Efficiency Benchmark.
    Inherits from MBART class (which does most of the processing).
    The main difference is how model + tokenizer is loaded.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        task: str,
        quantize_mode: str,
        max_bsz: int = 32,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path, task, quantize_mode, max_bsz
        )

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
