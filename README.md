# LSRA: A Layer-Selective Representation Alignment Framework for Unlearning in MLLMs

LSRA is a multimodal machine unlearning framework that removes target
knowledge consistently from both text-only and image-text inputs while
preserving the utility of retained knowledge. It first selects layers that are
effective for forgetting with limited retention degradation, then aligns
hidden representations at the selected layers using forget and retain targets.

## Overview

Multimodal large language models (MLLMs) can expose the same target knowledge
through different input modalities. For example, a fact may still be recovered
through a text-only query even after it has been removed from an image-text
query. Unlearning updates can also damage unrelated retained knowledge.

LSRA addresses these issues in three stages:

1. **Layer selection.** Candidate layers are evaluated jointly on unimodal and
   multimodal forget/retain data. A distance constraint encourages selected
   layers to be distributed across the network depth.
2. **Target construction.** Forget targets are extracted from a frozen
   reference model using a shared random-token sequence. Retain targets are
   reference-model representations of retain samples.
3. **Representation alignment.** Forget representations are redirected toward
   forget targets, while retain representations are regularized toward their
   original states. Only selected MLP `down_proj` weights are updated.

<p align="center">
  < img src="figure/framework.png" alt="Overview of LSRA" width="92%">
</p >

## Installation

Our experiments were conducted with Python 3.10 and PyTorch built for
CUDA 12.1. The core dependencies in our environment are:

| Package | Version |
| --- | --- |
| Python | 3.10.4 |
| PyTorch | 2.4.0+cu121 |
| Transformers | 4.57.3 |
| Accelerate | 1.7.0 |
| PEFT | 0.15.2 |
| Datasets | 2.21.0 |
| LLaMA-Factory | 0.9.3 |
| Safetensors | 0.4.4 |
| NumPy | 1.26.4 |
| Pillow | 10.4.0 |
| scikit-learn | 1.5.2 |
| Rouge | 1.0.1 |

The LLaMA-Factory baseline configurations under `llamafactory/` require an
installed [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
environment. Other transitive or visualization dependencies can be installed
as required by the corresponding evaluation and plotting scripts.

## Quick Start

### 1. Layer Selection

Run mini-training over candidate transformer layers, rank their
forgetting-retention trade-off, and select dispersed layers:

```bash
python utils/layer_selector.py \
    --model Qwen2.5-VL-3B \
    --dataset umu \
    --data_dir ./datasets \
    --forget_ratio 5 \
    --start_layer 0 \
    --end_layer 35 \
    --num_steps 500 \
    --top_k 4 \
    --min_gap 5 \

``` 

### 2. Representation Alignment for Unlearning


```bash
python methods/LSRA.py \
    --model Qwen2.5-VL-3B \
    --dataset umu \
    --data_dir ./datasets \
    --forget_ratio 5 \
    --target_layers 0 9 15 22 \
    --trainable_params_regex "model.language_model.layers.(0|9|15|22).mlp.down_proj.weight" \
    --loss_type cosine \
    --random_seq_len 1024 \
    --batch_size 2 \
    --num_epochs 9 \
    --learning_rate 1e-4 \
    --model_device cuda:0 \
```



### 3. Evaluation

Evaluate a trained checkpoint as follows:

```bash
python eval/run_eval.py \
    --model Qwen2.5-VL-3B \
    --model_path ./outputs/lsra/<checkpoint_dir> \
    --dataset umu \
    --base_path ./datasets \
    --batch_size 1 \
    --model_device cuda:0 \
    --judge_device cuda:1
```


## Main Results

Results below are reported in the paper for LSRA. Lower forget-set metrics
(`F*`) are better, while higher retain-set metrics (`R*`) and H-Mean are
better.

| Model | Dataset | FVQA | RVQA | FVGEN | RVGEN | FQA/FGEN | RQA/RGEN | H-Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | UMU-Bench | 0.0133 | 0.6367 | 0.0243 | 0.6065 | 0.0200 | 0.7400 | 0.6747 |
| Qwen2.5-VL-3B-Instruct | CLEAR | 0.1117 | 0.8734 | 0.1555 | 0.5034 | 0.0055 | 0.8804 | 0.6396 |
| LLaVA-1.5-7B | UMU-Bench | 0.0200 | 0.7482 | 0.0075 | 0.9510 | 0.0168 | 0.6979 | 0.8070 |
| LLaVA-1.5-7B | CLEAR | 0.0638 | 0.5189 | 0.0832 | 0.8623 | 0.0096 | 0.9758 | 0.6780 |

For UMU-Bench, the text-only columns correspond to FQA/RQA; for CLEAR, they
correspond to FGEN/RGEN.
