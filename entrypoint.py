import argparse
import json
import os
import sys

import transformers
from datasets import Dataset

from example_stdio_submission_sst import GoodBinarySentimentClassifier
from mbart import MBART
from t5 import T5


# We provide this
def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    for line in sys.stdin:
        line = line.rstrip()
        inputs = json.loads(line)
        assert isinstance(inputs, list)
        # Participants need to connect their inference code to our wrapper through the following line.
        outputs = predictor.predict(inputs=inputs)
        # Writes are \n deliminated, so adding \n is essential to separate this write from the next loop iteration.
        # outputs = [o for o in outputs]
        outputs = list(outputs)
        sys.stdout.write(f"{json.dumps(outputs)}\n")
        # Writes to stdout are buffered. The flush ensures the output is immediately sent through the pipe
        # instead of buffered.
        sys.stdout.flush()


def offline_predictor_wrapper(predictor: MBART):
    configs = sys.stdin.readline().rstrip()
    configs = json.loads(configs)
    assert isinstance(configs, dict)

    offline_dataset = Dataset.from_json(configs["offline_data_path"])
    offline_dataset_inputs = [instance["input"] for instance in offline_dataset]
    predictor.prepare()
    sys.stdout.write("Model and data loaded. Start the timer.\n")
    sys.stdout.flush()
    
    limit = configs.get("limit", None)
    if limit is not None and limit > 0:
        offline_dataset_inputs = offline_dataset_inputs[:limit]
    outputs = predictor.predict_offline(offline_dataset_inputs)
    outputs = list(outputs)
    sys.stdout.write("Offiline prediction done. Stop the timer.\n")
    sys.stdout.flush()

    outputs = Dataset.from_list([{"output": o} for o in outputs])
    outputs.to_json(configs["offline_output_path"])
    sys.stdout.write("Offiline outputs written. Exit.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    # We read outputs from stdout, and it is crucial to surpress unnecessary logging to stdout
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    transformers.logging.disable_progress_bar()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if "t5" in args.model:
        predictor = T5(
            pretrained_model_name_or_path=args.model,
            task=args.task
        )
    elif args.model == "mbart":
        predictor = MBART()
    elif args.model == "debug":
        predictor = GoodBinarySentimentClassifier()
    else:
        raise NotImplementedError()
    if args.offline:
        offline_predictor_wrapper(predictor)
    else:   
        stdio_predictor_wrapper(predictor)
