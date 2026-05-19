<div align="center">

# CRM: Coherent Representation Misdirection for Multimodal Unlearning

<p>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-ACL%202026-red.svg" alt="Paper"></a>
</p>

<p><i>A representation-level framework for consistent and utility-preserving unlearning in multimodal large language models.</i></p>

</div>

---

## Overview

Multimodal Large Language Models (MLLMs) absorb sensitive or outdated knowledge during pretraining, and removing such knowledge without retraining is a critical challenge. **CRM (Coherent Representation Misdirection)** formulates MLLM unlearning as a *representation redirection* problem and addresses two key issues that existing methods struggle with:

- **Cross-modal inconsistency.** Target knowledge can be accessed through *text-only* or *image-text* queries, but most methods only erase it from one side.
- **Forgetting vs. utility trade-off.** Aggressive updates often hurt the model's general capabilities.

CRM tackles both with a two-stage design:

1. **Layer Selection** &nbsp;—&nbsp; Identifies layers that contribute strongly to forgetting while introducing minimal degradation to retention, using a *joint* unimodal + multimodal score and a *distance constraint* that disperses selected layers across depths.
2. **Coherent Representation Misdirection** &nbsp;—&nbsp; On the selected layers, forget-sample activations are pushed toward **coherent targets** derived from a single forward pass of the original model on a shared random token sequence, while retain-sample activations are kept close to the original states. Only the **MLP down-projection** weights of selected layers are updated.

<div align="center">
  <img src="figure/framework.png" alt="CRM framework" width="92%">
  <br>
  <em>Figure 1. Overview of CRM. Stage 1 selects forgetting-critical layers; Stage 2 redirects forget representations toward coherent random targets while preserving retain representations.</em>
</div>

---

## Repository Structure

```
CRM/
├── methods/                # Core CRM training code
│   ├── CRM.py              # Main unlearning entry point (Stage 2)
│   └── load_data.py        # Forget / retain dataloaders
├── utils/                  # Helpers
│   ├── layer_selector.py   # Stage 1: one-pass layer selection
│   ├── model_utils.py      # Model loading, LoRA, hooks
│   ├── layer_utils.py
│   └── seed_utils.py
├── eval/                   # Evaluation pipeline
│   ├── run_eval.py         # Main evaluation entry point
│   ├── evaluate.py         # FVQA / FVGEN / FQA / RVQA / RVGEN / RQA
│   └── score_by_llm.py     # LLM-as-judge scoring
├── metrics/                # ROUGE / BLEU / BERTScore / Accuracy / ...
├── llamafactory/           # LLaMA-Factory configs for baseline fine-tuning
├── figure/                 # Framework figure
└── README.md
```

---

## Installation

> **Note**: The package versions below are placeholders — please replace with the versions used in your environment once the server is back online.

```bash
git clone https://github.com/<your-org>/CRM-Unlearning.git
cd CRM-Unlearning

conda create -n crm python=3.10 -y
conda activate crm

pip install -r requirements.txt
```

<details>
<summary><b>Main dependencies</b></summary>

- `torch >= 2.1`
- `transformers >= 4.45`
- `accelerate`
- `peft`
- `datasets`
- `Pillow`, `numpy`, `scikit-learn`, `tqdm`
- `evaluate` (for ROUGE / BLEU / BERTScore)

</details>

---

## Data & Models

### Datasets

| Dataset | Modalities | Forget Ratios | Link |
|---|---|---|---|
| **UMU-Bench** | Text + Image-Text | 5% / 10% / 15% | [link](#) |
| **CLEAR** | Text + Image-Text | 1% / 5% / 10% | [link](#) |

After downloading, place the datasets under `./datasets/` (or pass `--data_dir` explicitly).

### Pretrained MLLMs

| Model | HF Hub |
|---|---|
| Qwen2.5-VL-3B-Instruct | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| LLaVA-1.5-7B | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |

---

## Quick Start

### Stage 1 — Layer Selection

Rank all transformer layers by their joint forgetting / retention scores and dump a layer-ranking JSON.

```bash
python utils/layer_selector.py \
    --model LLaVA-1.5-7B \
    --dataset clear \
    --start_layer 0 \
    --end_layer 31 \
    --num_steps 500 \
    --learning_rate 1e-4
```

Output: `outputs/layer_select/{dataset}_{model}/layer_ranking.json`

### Stage 2 — Coherent Representation Misdirection

Run unlearning on the layers selected in Stage 1 (here we use `0 9 14 19` as an example):

```bash
python methods/CRM.py \
    --model LLaVA-1.5-7B \
    --dataset clear \
    --forget_ratio 5 \
    --target_layers 0 9 14 19 \
    --loss_type cosine \
    --num_epochs 9 \
    --learning_rate 1e-4 \
    --gamma 1.0 \
    --alpha 1.0 \
    --random_seq_len 1024 \
    --output_dir ./outputs/CRM
```

### Evaluation

```bash
python eval/run_eval.py \
    --model LLaVA-1.5-7B \
    --model_path ./outputs/CRM/<your-checkpoint-dir> \
    --dataset clear \
    --batch_size 1
```

The script reports **FVQA / FVGEN / FQA** (forget, ↓), **RVQA / RVGEN / RQA** (retain, ↑) and the overall **H-Mean** (↑).

---

## Key Hyperparameters

| Argument | Description | Default |
|---|---|---|
| `--target_layers` | Layer indices selected by Stage 1 | `0 9 14 19` |
| `--random_seq_len` | Length of the shared random token sequence | `1024` |
| `--loss_type` | `cosine` (recommended) or `mse` | `cosine` |
| `--gamma` | Weight of the forget loss | `1.0` |
| `--alpha` | Weight of the retain loss | `1.0` |
| `--text_weight` / `--multimodal_weight` | Modality weights $w_t$ / $w_m$ | `1.0` / `1.0` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--num_epochs` | Training epochs | `9` |

Only the **MLP `down_proj`** weights of the selected layers are trainable, as controlled by `--trainable_params_regex`.

---

## Main Results

Selected results from the paper (full table in Section 4.3):

| Model | Method | FVQA ↓ | RVQA ↑ | FVGEN ↓ | RVGEN ↑ | FQA ↓ | RQA ↑ | **H-Mean ↑** |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-VL-3B** | Vanilla | 72.00% | 62.84% | 0.6672 | 0.6716 | 76.00% | 72.21% | – |
|  | MIP-Editor | 4.80% | 61.84% | 0.0997 | 0.6564 | 9.60% | 61.81% | 0.6306 |
|  | **CRM (Ours)** | **1.33%** | **63.67%** | **0.0243** | 0.6065 | **2.00%** | **74.00%** | **0.6747** |
| **LLaVA-1.5-7B** | Vanilla | 76.00% | 72.21% | 0.9900 | 0.9702 | 82.00% | 69.05% | – |
|  | MIP-Editor | 38.40% | 67.87% | 0.3418 | 0.8308 | 36.80% | 63.80% | 0.5629 |
|  | **CRM (Ours)** | **2.00%** | **74.82%** | **0.0075** | **0.9510** | **1.68%** | **69.79%** | **0.8070** |

CRM achieves consistent low forget scores across all modalities while preserving the highest retain performance.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{anonymous2026crm,
  title     = {Coherent Representation Misdirection for Multimodal Unlearning},
  author    = {Anonymous},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon many great open-source projects, including
[UMU-Bench](#), [CLEAR](#), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the HuggingFace `transformers` / `peft` / `accelerate` ecosystem.

---

## License

This project is released under the [MIT License](LICENSE).
