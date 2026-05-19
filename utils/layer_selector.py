"""
One-pass layer selector for Multi-Layer RMU.

Loads the model once, mini-trains each candidate layer independently for N
steps (then restores original weights), records per-step forget/retain losses
(uf, mf, ur, mr), and ranks layers using a ratio-aware formula.

Replaces the old two-step workflow:
  1. run_multilayer_rmu_sweep.py   (subprocess per layer, very slow)
  2. rank_layers_from_loss_step.py (post-hoc ranking from json files)

Usage:
  python utils/layer_selector.py --model LLaVA-1.5-7B --dataset clear --start_layer 0 --end_layer 31 --num_steps 500 --learning_rate 1e-4

Output:
  outputs/layer_select/{dataset}_{model}/layer_ranking.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import copy
import json
import sys
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "methods", "baselines"))

from methods.baselines.load_data import (
    load_forget_dataloaders,
    load_retain_dataloaders,
)
from methods.baselines.multilayer_rmu import (
    RandomTargetCache,
    build_answer_mask_for_multimodal,
    compute_activation_loss,
    compute_activation_loss_cosine,
    forward_with_cache_multilayer,
    get_layer_modules,
)
from utils.model_utils import get_supported_models, load_model_and_processor
from utils.seed_utils import set_seed

# ── helpers ──────────────────────────────────────────────────────────────────


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _get_language_layers(model):
    raw = _unwrap(model)
    if hasattr(raw, "base_model") and hasattr(raw.base_model, "model"):
        raw = raw.base_model.model
    lm = getattr(raw, "language_model", None)
    if lm is None:
        raise ValueError("Cannot find language_model in the model")
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    raise ValueError("Cannot find transformer layers in language_model")


def _get_target_weight(model, layer_idx: int) -> nn.Parameter:
    return _get_language_layers(model)[layer_idx].mlp.down_proj.weight


def _build_inputs(batch: Dict) -> Dict:
    """Build model-input dict (no labels, no use_cache)."""
    inputs: Dict = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if "pixel_values" in batch:
        inputs["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        inputs["image_grid_thw"] = batch["image_grid_thw"]
    elif "grid_thw" in batch:
        inputs["image_grid_thw"] = batch["grid_thw"]
    return inputs


def _move_batch(batch: Dict, device) -> Dict:
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


# ── single-layer loss helpers ────────────────────────────────────────────────


def _forget_loss_one_layer(
    model, batch, module, target_cache, cache_idx, loss_type
):
    inputs = _build_inputs(batch)
    acts, _ = forward_with_cache_multilayer(
        model, inputs, [module], no_grad=False
    )
    act = acts[0]
    mask = build_answer_mask_for_multimodal(
        batch["input_ids"], batch["labels"], act.shape[1], act.device,
    )
    bs = batch["input_ids"].shape[0]
    tgt = target_cache.get_target(cache_idx, bs, act.shape[1])
    tgt = tgt.to(dtype=act.dtype, device=act.device)
    fn = (
        compute_activation_loss_cosine
        if loss_type == "cosine"
        else compute_activation_loss
    )
    return fn(act, tgt, mask)


def _retain_loss_one_layer(
    model, ref_model, batch, module, ref_module, loss_type
):
    inputs = _build_inputs(batch)
    if "labels" in batch:
        inputs["labels"] = batch["labels"]
    acts, _ = forward_with_cache_multilayer(
        model, inputs, [module], no_grad=False
    )
    with torch.no_grad():
        ref_acts, _ = forward_with_cache_multilayer(
            ref_model, inputs, [ref_module], no_grad=True
        )
    act, ref_act = acts[0], ref_acts[0]
    mask = build_answer_mask_for_multimodal(
        batch["input_ids"], batch["labels"], act.shape[1], act.device,
    )
    fn = (
        compute_activation_loss_cosine
        if loss_type == "cosine"
        else compute_activation_loss
    )
    return fn(act, ref_act.to(act.device), mask)


# ── data pre-collection ──────────────────────────────────────────────────────


def collect_batches(dataloaders, num_steps, device):
    """Pre-collect *num_steps* batches per data source and pin them on *device*.

    Returns (collected_dict, actual_steps).
    """
    keys = ["multi_forget", "uni_forget", "multi_retain", "uni_retain"]
    collected: Dict[str, list] = {k: [] for k in keys}
    for k in keys:
        for i, batch in enumerate(dataloaders[k]):
            if i >= num_steps:
                break
            collected[k].append(_move_batch(batch, device))
    actual = min(len(collected[k]) for k in keys)
    return collected, actual


# ── core: sweep one layer (mini-train → record losses → restore) ─────────


def sweep_one_layer(
    model,
    ref_model,
    layer_idx,
    cache_idx,
    model_module,
    ref_module,
    target_cache,
    batches,
    num_steps,
    lr,
    loss_type,
    gamma,
    alpha,
    text_w,
    multi_w,
):
    target_weight = _get_target_weight(model, layer_idx)
    backup = target_weight.data.clone()

    for p in model.parameters():
        p.requires_grad = False
    target_weight.requires_grad = True
    optimizer = torch.optim.AdamW([target_weight], lr=lr)

    records: List[Dict] = []
    for step in range(num_steps):
        uf = _forget_loss_one_layer(
            model, batches["uni_forget"][step],
            model_module, target_cache, cache_idx, loss_type,
        )
        mf = _forget_loss_one_layer(
            model, batches["multi_forget"][step],
            model_module, target_cache, cache_idx, loss_type,
        )
        ur = _retain_loss_one_layer(
            model, ref_model, batches["uni_retain"][step],
            model_module, ref_module, loss_type,
        )
        mr = _retain_loss_one_layer(
            model, ref_model, batches["multi_retain"][step],
            model_module, ref_module, loss_type,
        )

        total = (
            gamma * (text_w * uf + multi_w * mf)
            + alpha * (text_w * ur + multi_w * mr)
        )
        total.backward()
        optimizer.step()
        optimizer.zero_grad()

        records.append({
            "step": step + 1,
            "uf": float(uf.item()),
            "mf": float(mf.item()),
            "ur": float(ur.item()),
            "mr": float(mr.item()),
            "total": float(total.item()),
        })

    target_weight.data.copy_(backup)
    target_weight.requires_grad = False
    del optimizer, backup
    torch.cuda.empty_cache()
    return records



# ── per-layer loss file (backward compatible) ────────────────────────────────


def _save_per_layer_loss(args, layer_idx: int, records: List[Dict]):
    """Save one layer's loss records to outputs/loss/{dataset}_{model}/{layer}.json

    Format is compatible with the old rank_layers_from_loss_step.py reader.
    """
    safe_name = args.model.replace("/", "_").replace(" ", "_")
    folder = os.path.join("outputs", "loss", f"{args.dataset}_{safe_name}")
    os.makedirs(folder, exist_ok=True)
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "target_layers": [layer_idx],
        "records": records,
    }
    path = os.path.join(folder, f"{layer_idx}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── ranking ──────────────────────────────────────────────────────────────────


def min_max_normalize(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_layers(
    layer_records: Dict[int, List[Dict]],
    num_avg_steps: int,
    text_w: float,
    multi_w: float,
) -> List[Dict]:
    """Rank layers by difference score.

    text_score  = Norm(uf) - Norm(ur)
    multi_score = Norm(mf) - Norm(mr)
    score = Norm( tw * text_score + mw * multi_score )

    We use a difference (not a ratio) to avoid blow-ups when the retain loss
    is close to zero. A larger difference means a larger contribution to
    forgetting with smaller damage to retention, so the layer is more
    suitable for editing.
    """
    layers = sorted(layer_records.keys())

    uf_list, mf_list, ur_list, mr_list = [], [], [], []
    for layer in layers:
        recs = layer_records[layer][:num_avg_steps]
        n = max(len(recs), 1)
        uf_list.append(sum(r["uf"] for r in recs) / n)
        mf_list.append(sum(r["mf"] for r in recs) / n)
        ur_list.append(sum(r["ur"] for r in recs) / n)
        mr_list.append(sum(r["mr"] for r in recs) / n)

    uf_n = min_max_normalize(uf_list)
    mf_n = min_max_normalize(mf_list)
    ur_n = min_max_normalize(ur_list)
    mr_n = min_max_normalize(mr_list)

    tw = text_w / (text_w + multi_w)
    mw = multi_w / (text_w + multi_w)

    text_diff = [uf_n[i] - ur_n[i] for i in range(len(layers))]
    multi_diff = [mf_n[i] - mr_n[i] for i in range(len(layers))]
    combined = [tw * text_diff[i] + mw * multi_diff[i]
                for i in range(len(layers))]
    score_list = min_max_normalize(combined)

    scores = []
    for i, layer in enumerate(layers):
        scores.append({
            "layer": layer,
            "raw": {
                "uf": uf_list[i], "mf": mf_list[i],
                "ur": ur_list[i], "mr": mr_list[i],
            },
            "norm_uf": uf_n[i], "norm_mf": mf_n[i],
            "norm_ur": ur_n[i], "norm_mr": mr_n[i],
            "text_diff": text_diff[i],
            "multi_diff": multi_diff[i],
            "score": score_list[i],
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def greedy_select(ranked: List[Dict], top_k: int, min_gap: int) -> List[Dict]:
    """Greedy layer selection with distance penalty.

    1. Pick the highest-score layer
    2. For remaining layers, penalize by proximity to already-selected layers:
       adjusted_score = score * min(dist_to_nearest_selected / min_gap, 1.0)
    3. Pick the highest adjusted_score, repeat until top_k layers selected
    """
    if min_gap <= 0 or top_k <= 1:
        return ranked[:top_k]

    score_map = {e["layer"]: e["score"] for e in ranked}
    remaining = set(score_map.keys())
    selected: List[int] = []

    for _ in range(min(top_k, len(ranked))):
        best_layer = None
        best_adj = -1.0
        for layer in remaining:
            if not selected:
                adj = score_map[layer]
            else:
                dist = min(abs(layer - s) for s in selected)
                penalty = min(dist / min_gap, 1.0)
                adj = score_map[layer] * penalty
            if adj > best_adj:
                best_adj = adj
                best_layer = layer
        selected.append(best_layer)
        remaining.remove(best_layer)

    layer_to_entry = {e["layer"]: e for e in ranked}
    return [layer_to_entry[l] for l in selected]


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="One-pass layer selector for Multi-Layer RMU",
    )

    g = p.add_argument_group("Model")
    g.add_argument("--model", type=str, default="LLaVA-1.5-7B",
                   choices=get_supported_models())
    g.add_argument("--model_path", type=str, default=None)
    g.add_argument("--model_device", type=str, default="cuda:0")
    g.add_argument("--ref_device", type=str, default='cuda:0',
                   help="Device for frozen ref model (default: auto-select "
                        "another GPU, or cpu if single GPU)")

    g = p.add_argument_group("Data")
    g.add_argument("--dataset", type=str, default="umu",
                   choices=["umu", "clear"])
    g.add_argument("--data_dir", type=str, default="./datasets")
    g.add_argument("--forget_ratio", type=int, default=5)
    g.add_argument("--image_resize", type=int, default=224)
    g.add_argument("--batch_size", type=int, default=2)
    g.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("Sweep")
    g.add_argument("--start_layer", type=int, default=0)
    g.add_argument("--end_layer", type=int, default=35)
    g.add_argument("--num_steps", type=int, default=100,
                   help="Mini-training steps per layer")
    g.add_argument("--learning_rate", type=float, default=1e-4)
    g.add_argument("--random_seq_len", type=int, default=1024)
    g.add_argument("--loss_type", type=str, default="cosine",
                   choices=["mse", "cosine"])
    g.add_argument("--gamma", type=float, default=1.0,
                   help="Forget loss weight during mini-training")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="Retain loss weight during mini-training")
    g.add_argument("--text_weight", type=float, default=1.0,
                   help="Text modality weight during mini-training")
    g.add_argument("--multi_weight", type=float, default=1.0,
                   help="Multimodal weight during mini-training")

    g = p.add_argument_group("Ranking")
    g.add_argument("--num_avg_steps", type=int, default=None,
                   help="Number of initial steps to average (default: all)")
    g.add_argument("--rank_text_weight", type=float, default=0.5)
    g.add_argument("--rank_multi_weight", type=float, default=0.5)
    g.add_argument("--top_k", type=int, default=30,
                   help="Number of top layers to recommend")
    g.add_argument("--min_gap", type=int, default=5,
                   help="Min distance between selected layers for diversity "
                        "(0 = no penalty, greedy disabled)")

    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", type=str,
                   default="outputs/layer_select")
    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.num_avg_steps is None:
        args.num_avg_steps = args.num_steps

    # restrict visible GPUs before any CUDA initialization
    model_gpu = int(str(args.model_device).replace("cuda:", ""))
    if args.ref_device and args.ref_device.startswith("cuda:"):
        ref_gpu = int(args.ref_device.replace("cuda:", ""))
        visible = sorted(set([model_gpu, ref_gpu]))
    else:
        visible = [model_gpu]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in visible)
    # remap device indices after CUDA_VISIBLE_DEVICES
    args.model_device = f"cuda:{visible.index(model_gpu)}"
    if args.ref_device and args.ref_device.startswith("cuda:"):
        args.ref_device = f"cuda:{visible.index(ref_gpu)}"

    print("=" * 70)
    print(" Layer Selector for Multi-Layer RMU")
    print(f" CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print("=" * 70)

    # ── 1. model ──
    print(f"\n[1/5] Loading model: {args.model} → {args.model_device}")
    model, processor = load_model_and_processor(
        model_name=args.model,
        model_path=args.model_path,
        model_device=args.model_device,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.resize_token_embeddings(len(processor.tokenizer))

    # ── 2. ref model (deepcopy on CPU to avoid GPU memory spike) ──
    if args.ref_device is None:
        if torch.cuda.device_count() > 1:
            model_gpu = int(str(args.model_device).replace("cuda:", ""))
            ref_gpu = 1 if model_gpu != 1 else 0
            args.ref_device = f"cuda:{ref_gpu}"
        else:
            args.ref_device = "cpu"
    print(f"[2/5] Creating frozen reference model on {args.ref_device} ...")
    model.cpu()
    ref_model = copy.deepcopy(model)
    model.to(args.model_device)
    ref_model.to(args.ref_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── 3. data ──
    print(f"[3/5] Loading data ({args.dataset}) ...")
    loader_mf, loader_uf = load_forget_dataloaders(
        processor=processor, args=args,
    )
    loader_mr, loader_ur = load_retain_dataloaders(
        processor=processor, args=args,
    )
    dataloaders = {
        "multi_forget": loader_mf, "uni_forget": loader_uf,
        "multi_retain": loader_mr, "uni_retain": loader_ur,
    }

    batches, actual_steps = collect_batches(
        dataloaders, args.num_steps, args.model_device,
    )
    if actual_steps < args.num_steps:
        print(f"  Warning: only {actual_steps} batches available, "
              f"reducing num_steps from {args.num_steps}")
        args.num_steps = actual_steps
    print(f"  Collected {actual_steps} batches per data source")

    # ── 4. target cache ──
    candidate_layers = list(range(args.start_layer, args.end_layer + 1))
    num_layers = len(candidate_layers)
    print(f"[4/5] Building random-token target cache "
          f"for {num_layers} layers ...")
    model_modules = get_layer_modules(model, candidate_layers)
    ref_modules = get_layer_modules(ref_model, candidate_layers)
    target_cache = RandomTargetCache(
        ref_model=ref_model,
        ref_modules=ref_modules,
        vocab_size=len(processor.tokenizer),
        seq_len=args.random_seq_len,
        device=args.model_device,
        dtype=next(ref_model.parameters()).dtype,
    )

    # ── 5. sweep ──
    print(f"[5/5] Sweeping layers {args.start_layer}..{args.end_layer} "
          f"({args.num_steps} steps, lr={args.learning_rate})\n")
    model.train()

    layer_records: Dict[int, List[Dict]] = {}

    for i, layer in enumerate(candidate_layers):
        t0 = datetime.now()

        records = sweep_one_layer(
            model, ref_model, layer, cache_idx=i,
            model_module=model_modules[i],
            ref_module=ref_modules[i],
            target_cache=target_cache,
            batches=batches,
            num_steps=args.num_steps,
            lr=args.learning_rate,
            loss_type=args.loss_type,
            gamma=args.gamma, alpha=args.alpha,
            text_w=args.text_weight, multi_w=args.multi_weight,
        )
        layer_records[layer] = records

        _save_per_layer_loss(args, layer, records)

        elapsed = (datetime.now() - t0).total_seconds()
        last = records[-1]
        print(
            f"  Layer {layer:3d}  ({elapsed:5.1f}s) "
            f"uf={last['uf']:.4f}  mf={last['mf']:.4f}  "
            f"ur={last['ur']:.4f}  mr={last['mr']:.4f}"
        )

    # ── 6. rank ──
    ranked = rank_layers(
        layer_records,
        args.num_avg_steps,
        args.rank_text_weight,
        args.rank_multi_weight,
    )
    selected = greedy_select(ranked, args.top_k, args.min_gap)

    # ── 7. save ──
    safe_name = args.model.replace("/", "_").replace(" ", "_")
    save_dir = os.path.join(
        args.output_dir, f"{args.dataset}_{safe_name}",
    )
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "layer_ranking.json")

    output = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "start_layer": args.start_layer,
            "end_layer": args.end_layer,
            "num_steps": args.num_steps,
            "num_avg_steps": args.num_avg_steps,
            "learning_rate": args.learning_rate,
            "loss_type": args.loss_type,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "text_weight": args.text_weight,
            "multi_weight": args.multi_weight,
            "rank_text_weight": args.rank_text_weight,
            "rank_multi_weight": args.rank_multi_weight,
            "top_k": args.top_k,
            "min_gap": args.min_gap,
        },
        "formula": {
            "score": "Norm( tw * (Norm(uf) - Norm(ur)) + mw * (Norm(mf) - Norm(mr)) )",
            "greedy": "adjusted = score * min(dist_to_nearest_selected / min_gap, 1.0)",
        },
        "ranking_all": ranked,
        "selected": selected,
        "per_layer_records": {
            str(k): v for k, v in layer_records.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    loss_dir = os.path.join("outputs", "loss", f"{args.dataset}_{safe_name}")
    print(f"\nSaved per-layer loss files to: {loss_dir}/")
    print(f"Saved ranking to: {output_path}")

    # ── print summary ──
    k = len(selected)

    print(f"\n{'=' * 70}")
    print(f" Top-{k} layers by score (no diversity constraint)")
    print(f"{'=' * 70}")
    for entry in ranked[:k]:
        print(f"  Layer {entry['layer']:3d}  "
              f"score={entry['score']:.4f}  "
              f"(text_diff={entry['text_diff']:.4f}  "
              f"multi_diff={entry['multi_diff']:.4f})")

    print(f"\n{'=' * 70}")
    print(f" Selected {k} layers (greedy, min_gap={args.min_gap})")
    print(f"{'=' * 70}")
    for entry in selected:
        print(f"  Layer {entry['layer']:3d}  "
              f"score={entry['score']:.4f}  "
              f"(text_diff={entry['text_diff']:.4f}  "
              f"multi_diff={entry['multi_diff']:.4f})")

    recommended = [e["layer"] for e in selected]
    print(f"\nRecommended target_layers: {recommended}")
    print("=" * 70)


if __name__ == "__main__":
    main()
