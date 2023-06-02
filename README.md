Please refer to 

# Submission Example Repo

## Introduction
This is our example repository for submission guidelines to the efficiency benchmark. 
We focus on two task use cases here: `RAFT` and `WMT` for Seq2Seq MT efficiency evaluation.

- RAFT Evaluation: todo

- WMT: We provide an interface for `wmt14-en-de` (English to German), `wmt14-de-en` (German to English), 
    `wmt16-en-ro` (English to Romanian) and `wmt16-ro-en` (Romanian to English) tasks. 

We plan to add more tasks soon!

To install the driver code see https://github.com/allenai/efficiency-benchmark/tree/main.
 
## Repository

```
submission
├── Dockerfile                      # Spec for Docker runtime (do not modify)
├── README.md                       # Instructions
├── auto.py                         # Template for Seq2Seq Model submissions
├── entrypoint.py                   # Driver class for usage (see submission format below)
├── example_stdio_submission_sst.py # SST Similarity example
├── mbart.py                        # MBART model family example (for WMT tasks)
├── opus.py                         # OPUS model family example (for WMT tasks)
├── requirements.txt                # Requirements for runtime. You might need to change this
└── t5.py                           # T5 example
```

### Submission instructions

``` bash
git clone https://github.com/allenai/efficiency-benchmark.git
cd efficiency-benchmark/
pip install .   # We recommend using a new virtual environment
```

## To run the mBART WMT14-en-de example locally
```bash
https://github.com/haopeng-nlp/submission.git
cd submission
pip install -r requirements.txt # In the same (new) environment as above. These will be *your* requirements to run.
TASK=wmt14-en-de
SCENARIO=accuracy
MODEL="facebook/mbart-large-50-many-to-many-mmt"

# Everything after -- is your runtime command with specific runtime instructions for the Python script
efficiency-benchmark run --task $TASK  --max_batch_size 50 --scenario $SCENARIO \
    -- python entrypoint.py --model $MODEL --$TASK
```

## To submit to the benchmark server
```bash
TASK=wmt14-en-de
SCENARIO=accuracy
MODEL="facebook/mbart-large-50-many-to-many-mmt"

# Everything after -- is your runtime command with specific runtime instructions for the Python script
efficiency-benchmark submit --task $TASK  --max_batch_size 50 \
    -- python entrypoint.py --model $MODEL --$TASK
```

Please contact haop@ to get access to this machine.
