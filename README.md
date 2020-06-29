# ImprovedFlend

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone repo
cd reo
pip install -r requirements.txt
python -m spacy download en
```

## Installation with Docker

To install using docker please build the self-contained image:

```bash
docker build -t convai .
```

You can then enter the image  

```bash
ip-192-168-22-157:repo gladaikins$ docker run --rm -it convai bash
root@91e241bb823e:/# ls
Dockerfile  README.md  boot                  dev  home         lib    media  models  proc              root  sbin  sys  train.py  utils.py
LICENCE     bin        convai_evaluation.py  etc  interact.py  lib64  mnt    opt     requirements.txt  run   srv   tmp  usr       var
```

You can then run the `interact.py` script on the pretrained model:

```bash
python3 interact.py --model models/
```

The easiest way to download and use this model is just to run the `interact.py` script to talk with the model. Without any argument, this script will automatically download and cache our model.

## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python ./train.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=8 ./train.py  # Training on 8 GPUs
```

The training script accept several arguments to tweak the training:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
num_candidates | `int` | `2` | Number of candidates for training
max_history | `int` | `2` | Number of previous exchanges to keep in history
train_batch_size | `int` | `4` | Batch size for training
valid_batch_size | `int` | `4` | Batch size for validation
gradient_accumulation_steps | `int` | `8` | Accumulate gradients on several steps
lr | `float` | `6.25e-5` | Learning rate
lm_coef | `float` | `1.0` | LM loss coefficient
mc_coef | `float` | `1.0` | Multiple-choice loss coefficient
max_norm | `float` | `1.0` | Clipping gradient norm
n_epochs | `int` | `3` | Number of training epochs
personality_permutations | `int` | `1` | Number of permutations of personality sentences
device | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | Device (cuda or cpu)
fp16 | `str` | `""` | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)
local_rank | `int` | `-1` | Local rank for distributed training (-1: not distributed)

Here is how to reproduce our results on a server with 8 V100 GPUs (adapt number of nodes and batch sizes to your configuration):

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2
```

## Using the interaction script

The training script saves all the experiments and checkpoints in a sub-folder named with the timestamp of the experiment in the `./runs` folder of the repository base folder.

You can then use the interactive script to interact with the model simply by pointing to this folder.

Here is an example command line to run the interactive script:

```bash
python ./interact.py --model_checkpoint ./data/Apr17_13-31-38_thunder/  # run the interactive script with a training checkpoint
python ./interact.py  # run the interactive script with the finetuned model on our S3
```

The fine-tuned model will gives FINAL Hits@1: 0.715

The interactive script accept a few arguments to tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)
