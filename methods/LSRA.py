"""
Multi-layer RMU (Multi-Layer Representation Misdirection for Unlearning)

Core idea:
  Forget: use ref_model activations from random tokens at each layer as consistent targets,
  and align the current model to them on forget data.
  Retain: apply EMBED_DIFF at every target layer only on answer positions to prevent
  drift in any trainable layer.

Differences from multimodal_rmu.py:
  - Supports multiple target layers (module_regex can match multiple layers).
  - Forget targets are no longer random vectors, but real ref_model activations on
    random tokens, making targets consistent across layers.
  - Retain loss is computed only on answer positions without separately weighting image tokens.
"""

import argparse
import copy
import re
from datetime import datetime
import json
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from load_data import load_forget_dataloaders, load_retain_dataloaders
from utils.model_utils import MODELS, get_supported_models, load_model_and_processor
from utils.seed_utils import set_seed


def _build_loss_log_path(args) -> str:
    safe_model_name = str(args.model).replace("/", "_").replace(" ", "_")
    layer_tag = "_".join(str(x) for x in sorted(args.target_layers))
    folder = os.path.join("outputs", "loss", f"{args.dataset}_{safe_model_name}")
    return os.path.join(folder, f"{layer_tag}.json")


def _flush_loss_records(loss_log_path: str, args, loss_records: List[Dict]):
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "target_layers": sorted(args.target_layers),
        "records": loss_records,
    }
    with open(loss_log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==============================================================================
# Arguments
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Layer RMU Unlearning")

    # Model
    parser.add_argument("--model", type=str, default="LLaVA-1.5-7B",
                        choices=get_supported_models())
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_device", type=str, default="cuda:0")

    # Dataset
    parser.add_argument("--dataset", type=str, default="clear",
                        choices=["umu", "clear"])
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--forget_ratio", type=int, default=5)
    parser.add_argument("--image_resize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    # Multi-layer RMU parameters
    parser.add_argument("--target_layers", type=int, nargs="+",
                        default=[0,9,14,19],
                        help="目标层索引列表，会在这些层提取激活并计算损失")
    parser.add_argument("--trainable_params_regex", type=str, nargs="+",
                        default=["model.language_model.layers.(0|9|14|19).mlp.down_proj.weight"],
                        help="可训练参数的正则表达式列表")
    parser.add_argument("--random_seq_len", type=int, default=1024,
                        help="生成随机 token 序列的长度（用于遗忘目标）")

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=9)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="遗忘损失权重")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="保留损失权重")

    # Loss type
    parser.add_argument("--loss_type", type=str, default="cosine",
                        choices=["mse", "cosine"],
                        help="激活损失类型：mse 或 cosine（余弦距离，天然归一化）")

    # Loss normalization (only applies to MSE; cosine is naturally normalized)
    parser.add_argument("--normalize_loss", action="store_true", default=False,
                        help="对每层 MSE 损失除以该层激活均值的平方，消除模型间激活量级差异")

    # Modality weights
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--multimodal_weight", type=float, default=1.0)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/multilayer_rmu")
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="最大训练 step；-1 表示不限制")
    parser.add_argument("--save_model_weights", default= True,
                        help="为 true 时才保存模型和 processor 权重")

    return parser.parse_args()


# ==============================================================================
# Multi-layer hooks
# ==============================================================================

def get_layer_modules(model, layer_indices: List[int]) -> List[nn.Module]:
    """
    Get transformer layer modules by index.
    Automatically handles Accelerator / PeftModel wrappers.

    Returns:
        modules: Modules ordered by layer_indices.
    """
    # Handle Accelerator / DDP wrappers by stripping only the outermost .module.
    raw = model
    if hasattr(raw, 'module'):
        raw = raw.module

    pattern = re.compile(r"(?:.*\.)?language_model\.layers\.(\d+)$")
    name2mod = {}
    for name, mod in raw.named_modules():
        m = pattern.fullmatch(name)
        if m:
            name2mod[int(m.group(1))] = mod

    modules = []
    for idx in layer_indices:
        if idx not in name2mod:
            raise ValueError(
                f"Layer {idx} not found. Available: {sorted(name2mod.keys())}"
            )
        modules.append(name2mod[idx])
    return modules


def forward_with_cache_multilayer(
    model, inputs: Dict, modules: List[nn.Module], no_grad: bool = True
) -> Tuple[List[torch.Tensor], object]:
    """
    Run one forward pass while caching activations from multiple layers.

    Returns:
        activations: List with the same length as modules; each item has shape
            (batch, seq_len, hidden_dim).
        outputs: Full model output.
    """
    caches = [[] for _ in modules]

    def make_hook(idx):
        def hook_fn(module, inp, out):
            caches[idx].append(out[0] if isinstance(out, tuple) else out)
        return hook_fn

    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(modules)]

    with torch.set_grad_enabled(not no_grad):
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    activations = [c[0] for c in caches]
    return activations, outputs


# ==============================================================================
# Random-token target cache
# ==============================================================================

class RandomTargetCache:
    """
    Run random token sequences through ref_model and cache activations for each
    target layer.
    The same targets are reused throughout training because ref_model is frozen
    and deterministic.
    """

    def __init__(
        self,
        ref_model,
        ref_modules: List[nn.Module],
        vocab_size: int,
        seq_len: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.targets = self._build(
            ref_model, ref_modules, vocab_size, seq_len, device, dtype
        )

    @torch.no_grad()
    def _build(self, ref_model, ref_modules, vocab_size, seq_len, device, dtype):
        random_ids = torch.randint(100, 32000, (1, seq_len), device=device)
        attention_mask = torch.ones_like(random_ids)
        inputs = {"input_ids": random_ids, "attention_mask": attention_mask}

        activations, _ = forward_with_cache_multilayer(
            ref_model, inputs, ref_modules, no_grad=True
        )
        # Target for each layer: (1, seq_len, hidden_dim), detached and cached.
        return [act.detach() for act in activations]

    def get_target(self, layer_idx: int, batch_size: int, target_seq_len: int):
        """
        Get target activations for the layer_idx-th target layer.
        Use repeat plus truncation/padding to support different batch sizes and
        sequence lengths.

        Returns:
            target: (batch_size, target_seq_len, hidden_dim)
        """
        t = self.targets[layer_idx]  # (1, cache_seq_len, hidden_dim)
        cache_seq_len = t.shape[1]
        hidden_dim = t.shape[2]

        # Expand to batch_size.
        t = t.expand(batch_size, -1, -1)

        if target_seq_len <= cache_seq_len:
            return t[:, :target_seq_len, :]
        else:
            # Repeat cyclically to fill the requested sequence length.
            repeats = (target_seq_len // cache_seq_len) + 1
            t = t.repeat(1, repeats, 1)
            return t[:, :target_seq_len, :]


# ==============================================================================
# Loss functions
# ==============================================================================

def compute_activation_loss(
    activation1: torch.Tensor,
    activation2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Position-wise MSE loss with a mask.
    activation: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len), where True marks valid positions.
    """
    squared_diff = nn.functional.mse_loss(activation1, activation2, reduction="none")
    expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
    per_sample = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
    num_tokens = mask.sum(dim=-1).clamp(min=1)
    return (per_sample / num_tokens).mean()


def compute_activation_loss_cosine(
    activation1: torch.Tensor,
    activation2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Position-wise cosine distance loss (1 - cosine_similarity) with a mask.
    activation: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len), where True marks valid positions.

    Return range is [0, 2]: 0 = identical direction, 1 = orthogonal,
    2 = opposite direction.
    Naturally normalized and unaffected by activation scale.
    """
    cos_sim = nn.functional.cosine_similarity(activation1, activation2, dim=2)
    cos_dist = (1.0 - cos_sim).clamp(min=0.0)  # (batch, seq_len)  # (batch, seq_len)
    masked_dist = cos_dist * mask.float()
    num_tokens = mask.sum(dim=-1).clamp(min=1)
    per_sample = masked_dist.sum(dim=1) / num_tokens
    return per_sample.mean()


def build_answer_mask_for_multimodal(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    act_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build an answer mask for multimodal inputs that matches the activation
    sequence length.

    Background:
      - labels = input_ids.clone(), including image placeholder tokens (ID=32000).
      - answer mask = labels != -100, but image tokens should not contribute to loss.
      - act_len = sequence length of model_acts, the transformer layer output
        captured by the hook.

    Logic:
      1. answer mask = (labels != -100) & ~(input_ids == 32000), excluding image tokens.
      2. Align according to the relationship between act_len and answer_mask.shape[1]:
         - equal: return directly
         - text_len > act_len: truncate to the first act_len columns
         - text_len < act_len: pad the extra positions with False
    """
    # Special image token IDs (LLaVA / QwenVL).
    image_token_ids = [32000, 151655, 151652, 151653,151654,151656]
    is_image = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in image_token_ids:
        is_image = is_image | (input_ids == tid)

    # Answer mask: positions that are neither -100 nor image tokens.
    answer_mask = (labels != -100) & ~is_image  # (batch, text_seq_len)

    text_len = answer_mask.shape[1]

    if text_len == act_len:
        return answer_mask.to(device)
    elif text_len > act_len:
        # Truncate the extra positions.
        print("出现异常情况了1")
        return answer_mask[:, :act_len].to(device)
        print("特殊情况出现了")
    else:
        # Pad extra positions with False; these correspond to expanded image patches.
        print("出现异常情况了2")
        mask = torch.zeros(
            answer_mask.shape[0], act_len, dtype=torch.bool, device=device
        )
        mask[:, :text_len] = answer_mask
        print("特殊情况出现了")
        return mask


def compute_forget_loss(
    model,
    forget_batch: Dict,
    model_modules: List[nn.Module],
    target_cache: RandomTargetCache,
    modality: str,
    args,
) -> Tuple[torch.Tensor, Dict]:
    """
    Multi-layer forget loss: align forget-data activations to random-token
    targets at each target layer.

    Losses are averaged across layers.
    """
    inputs = {
        "input_ids": forget_batch["input_ids"],
        "attention_mask": forget_batch["attention_mask"],
    }
    if "pixel_values" in forget_batch:
        inputs["pixel_values"] = forget_batch["pixel_values"]
    if "image_grid_thw" in forget_batch:
        inputs["image_grid_thw"] = forget_batch["image_grid_thw"]
    elif "grid_thw" in forget_batch:
        inputs["image_grid_thw"] = forget_batch["grid_thw"]

    model_acts, _ = forward_with_cache_multilayer(
        model, inputs, model_modules, no_grad=False
    )

    # Align the mask by excluding image tokens and keeping only answer tokens.
    cur_mask = build_answer_mask_for_multimodal(
        forget_batch["input_ids"],
        forget_batch["labels"],
        model_acts[0].shape[1],
        model_acts[0].device,
    )
    batch_size = forget_batch["input_ids"].shape[0]

    targets = [
        target_cache.get_target(i, batch_size, act.shape[1]).to(dtype=act.dtype, device=act.device)
        for i, act in enumerate(model_acts)
    ]

    loss_fn = compute_activation_loss_cosine if args.loss_type == "cosine" else compute_activation_loss

    layer_losses = []
    layer_scales = []
    for act, tgt in zip(model_acts, targets):
        raw_loss = loss_fn(act, tgt, cur_mask)
        scale = (act.detach().abs().mean() ** 2).clamp(min=1e-8)
        layer_scales.append(scale.item())
        if args.loss_type == "mse" and args.normalize_loss:
            layer_losses.append(raw_loss / scale)
        else:
            layer_losses.append(raw_loss)

    forget_loss = torch.stack(layer_losses).mean()

    info = {
        "modality": modality,
        "num_layers": len(model_modules),
        "layer_losses": [l.item() for l in layer_losses],
        "num_answer_tokens": cur_mask.sum().item(),
        "act_mean": model_acts[0].abs().mean().item(),
        "target_mean": targets[0].abs().mean().item(),
        "cur_mask_sum": cur_mask.sum().item(),
        "norm_scales": layer_scales,
        "loss_type": args.loss_type,
    }
    return forget_loss, info


def compute_retain_loss(
    model,
    retain_batch: Dict,
    model_modules: List[nn.Module],
    ref_model,
    ref_modules: List[nn.Module],
    modality: str,
    args,
) -> Tuple[torch.Tensor, Dict]:
    """
    Multi-layer retain loss (EMBED_DIFF): at each target layer, keep the current
    model's retain-data activations close to the ref_model activations.

    Compute loss only on answer positions. When activation seq_len differs from
    input_ids because images are expanded, align to the valid part of answer_mask.
    """
    inputs = {
        "input_ids": retain_batch["input_ids"],
        "attention_mask": retain_batch["attention_mask"],
        "labels": retain_batch["labels"],
    }
    if "pixel_values" in retain_batch:
        inputs["pixel_values"] = retain_batch["pixel_values"]
    if "image_grid_thw" in retain_batch:
        inputs["image_grid_thw"] = retain_batch["image_grid_thw"]
    elif "grid_thw" in retain_batch:
        inputs["image_grid_thw"] = retain_batch["grid_thw"]

    model_acts, _ = forward_with_cache_multilayer(
        model, inputs, model_modules, no_grad=False
    )
    with torch.no_grad():
        ref_acts, _ = forward_with_cache_multilayer(
            ref_model, inputs, ref_modules, no_grad=True
        )

    # Align the mask by excluding image tokens and keeping only answer tokens.
    cur_mask = build_answer_mask_for_multimodal(
        retain_batch["input_ids"],
        retain_batch["labels"],
        model_acts[0].shape[1],
        model_acts[0].device,
    )

    loss_fn = compute_activation_loss_cosine if args.loss_type == "cosine" else compute_activation_loss

    layer_losses = []
    layer_scales = []
    for m_act, r_act in zip(model_acts, ref_acts):
        raw_loss = loss_fn(m_act, r_act.to(m_act.device), cur_mask)
        scale = (m_act.detach().abs().mean() ** 2).clamp(min=1e-8)
        layer_scales.append(scale.item())
        if args.loss_type == "mse" and args.normalize_loss:
            layer_losses.append(raw_loss / scale)
        else:
            layer_losses.append(raw_loss)

    retain_loss = torch.stack(layer_losses).mean()

    info = {
        "modality": modality,
        "num_layers": len(model_modules),
        "layer_losses": [l.item() for l in layer_losses],
        "num_answer_tokens": cur_mask.sum().item(),
        "norm_scales": layer_scales,
        "loss_type": args.loss_type,
    }
    return retain_loss, info


# ==============================================================================
# Data loading (reuses logic from multimodal_rmu)
# ==============================================================================

def load_model_and_data(args):
    print("\n" + "=" * 80)
    print("Step 1: 加载模型和数据集")
    print("=" * 80)

    print(f"\n加载模型: {args.model}")
    model, processor = load_model_and_processor(
        model_name=args.model,
        model_path=args.model_path,
        model_device=args.model_device,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.resize_token_embeddings(len(processor.tokenizer))

    loader_multi_forget, loader_uni_forget = load_forget_dataloaders(
        processor=processor, args=args
    )
    loader_multi_retain, loader_uni_retain = load_retain_dataloaders(
        processor=processor, args=args
    )

    dataloaders = {
        "multi_forget": loader_multi_forget,
        "uni_forget": loader_uni_forget,
        "multi_retain": loader_multi_retain,
        "uni_retain": loader_uni_retain,
    }
    return model, processor, dataloaders


# ==============================================================================
# Initialization
# ==============================================================================

def initialize_components(model, processor, args):
    print("\n" + "=" * 80)
    print("Step 2: 初始化多层 RMU 组件")
    print("=" * 80)

    target_layers = sorted(args.target_layers)
    print(f"\n1. 目标层: {target_layers}")

    model_modules = get_layer_modules(model, target_layers)
    print(f"  ✓ 找到 {len(model_modules)} 个目标层")

    # Reference model
    print(f"\n2. 创建参考模型")
    ref_model = copy.deepcopy(model).to(args.model_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ref_modules = get_layer_modules(ref_model, target_layers)
    print(f"  ✓ 参考模型创建完成 ({sum(p.numel() for p in ref_model.parameters()) / 1e9:.2f}B)")

    # Random-token target cache
    print(f"\n3. 生成随机 token 遗忘目标 (seq_len={args.random_seq_len})")
    vocab_size = len(processor.tokenizer)
    dtype = next(ref_model.parameters()).dtype
    target_cache = RandomTargetCache(
        ref_model=ref_model,
        ref_modules=ref_modules,
        vocab_size=vocab_size,
        seq_len=args.random_seq_len,
        device=args.model_device,
        dtype=dtype,
    )
    print(f"  ✓ 缓存了 {len(target_layers)} 层的目标激活")

    # Trainable parameter statistics
    print(f"\n4. 可训练参数: {args.trainable_params_regex}")
    count = 0
    names = []
    for name, param in model.named_parameters():
        if any(re.fullmatch(p, name) for p in args.trainable_params_regex):
            count += 1
            names.append(name)
    print(f"  ✓ 匹配 {count} 个参数")
    for n in names[:10]:
        print(f"    - {n}")
    if count > 10:
        print(f"    ... 还有 {count - 10} 个")

    print("=" * 80 + "\n")

    return {
        "model_modules": model_modules,
        "ref_model": ref_model,
        "ref_modules": ref_modules,
        "target_cache": target_cache,
        "trainable_params_regex": args.trainable_params_regex,
    }


# ==============================================================================
# Training loop
# ==============================================================================

def train_loop(
    model, processor, dataloaders, model_modules, ref_model, ref_modules,
    target_cache, accelerator, args, trainable_params_regex,
):
    print("\n" + "=" * 80)
    print("Step 3: 开始训练")
    print("=" * 80)

    # Freeze all parameters, then unfreeze only matched parameters.
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    for name, param in model.named_parameters():
        if any(re.fullmatch(p, name) for p in trainable_params_regex):
            param.requires_grad = True
            trainable_params.append(param)

    print(f"  可训练参数: {len(trainable_params)} 个, "
          f"{sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    optimizer = accelerator.prepare(optimizer)

    # Ensure again that only target parameters are trainable.
    for p in model.parameters():
        p.requires_grad = False
    for name, param in model.named_parameters():
        if any(re.fullmatch(p, name) for p in trainable_params_regex):
            param.requires_grad = True

    steps_per_epoch = min(
        len(dataloaders["multi_forget"]),
        len(dataloaders["uni_forget"]),
        len(dataloaders["multi_retain"]),
        len(dataloaders["uni_retain"]),
    )
    print(f"  每 epoch {steps_per_epoch} 步, 共 {args.num_epochs} epochs")
    print("=" * 80 + "\n")

    model.train()
    global_step = 0
    reached_max_steps = False
    loss_log_path = _build_loss_log_path(args)
    loss_records = []

    if accelerator.is_main_process:
        print(f"日志将保存到: {loss_log_path}")

    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'=' * 80}\n")

        combined = zip(
            dataloaders["multi_forget"],
            dataloaders["uni_forget"],
            dataloaders["multi_retain"],
            dataloaders["uni_retain"],
        )

        for batch_idx, (mf, uf, mr, ur) in enumerate(combined):
            step_start_time = time.perf_counter()

            if args.max_steps > 0 and global_step >= args.max_steps:
                reached_max_steps = True
                if accelerator.is_main_process:
                    print(f"\n达到 max_steps={args.max_steps}，提前停止训练。\n")
                break

            # Forget loss
            uni_fl, uni_fi = compute_forget_loss(
                model, uf, model_modules, target_cache, "text", args
            )
            multi_fl, multi_fi = compute_forget_loss(
                model, mf, model_modules, target_cache, "multimodal", args
            )

            # Retain loss
            uni_rl, uni_ri = compute_retain_loss(
                model, ur, model_modules, ref_model, ref_modules, "text", args
            )
            multi_rl, multi_ri = compute_retain_loss(
                model, mr, model_modules, ref_model, ref_modules, "multimodal", args
            )

            forget_w = args.text_weight * uni_fl + args.multimodal_weight * multi_fl
            retain_w = args.text_weight * uni_rl + args.multimodal_weight * multi_rl
            total_loss = args.gamma * forget_w + args.alpha * retain_w

            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            step_elapsed = time.perf_counter() - step_start_time

            if accelerator.is_main_process:
                log_record = {
                    "epoch": int(epoch + 1),
                    "step": int(batch_idx + 1),
                    "steps_per_epoch": int(steps_per_epoch),
                    "global_step": int(global_step),
                    "total": float(total_loss.item()),
                    "uf": float(uni_fl.item()),
                    "mf": float(multi_fl.item()),
                    "ur": float(uni_rl.item()),
                    "mr": float(multi_rl.item()),
                    "step_time_sec": float(step_elapsed),
                }
                loss_records.append(log_record)
                _flush_loss_records(loss_log_path, args, loss_records)

            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Step [{batch_idx+1}/{steps_per_epoch}] "
                    f"(Global: {global_step})"
                    f" | Total: {total_loss.item():.4f}"
                    f" | UF: {uni_fl.item():.4f}"
                    f" | MF: {multi_fl.item():.4f}"
                    f" | UR: {uni_rl.item():.4f}"
                    f" | MR: {multi_rl.item():.4f}"
                    f" | StepTime: {step_elapsed:.2f}s"
                )


        if reached_max_steps:
            break

    print(f"\n✓ 训练完成, 共 {global_step} 步\n")
    return global_step


# ==============================================================================
# Saving
# ==============================================================================

def save_checkpoint(model, processor, optimizer, step, epoch, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"{args.model}_{args.dataset}_epoch{epoch}_step{step}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n保存检查点到: {output_dir}")

    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

    config_dict = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            config_dict[k] = v
        except (TypeError, ValueError):
            config_dict[k] = str(v)
    config_dict.update({"step": step, "epoch": epoch, "timestamp": timestamp})

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"✓ 检查点保存完成\n")


# ==============================================================================
# Main function
# ==============================================================================

def main(args):
    set_seed(args.seed)
    accelerator = Accelerator()

    model, processor, dataloaders = load_model_and_data(args)
    components = initialize_components(model, processor, args)

    model = accelerator.prepare(model)
    for k in dataloaders:
        dataloaders[k] = accelerator.prepare(dataloaders[k])
    ref_model = accelerator.prepare(components["ref_model"])

    global_step = train_loop(
        model=model,
        processor=processor,
        dataloaders=dataloaders,
        model_modules=components["model_modules"],
        ref_model=ref_model,
        ref_modules=components["ref_modules"],
        target_cache=components["target_cache"],
        accelerator=accelerator,
        args=args,
        trainable_params_regex=components["trainable_params_regex"],
    )

    if accelerator.is_main_process:
        if args.save_model_weights:
            save_checkpoint(
                accelerator.unwrap_model(model), processor, None,
                global_step, args.num_epochs, args,
            )
        else:
            print("跳过最终模型保存（--save_model_weights 未开启）")
        print("\n" + "=" * 80)
        print("✓ 多层 RMU 训练完成！")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
