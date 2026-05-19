"""Shared helpers for multi-layer RMU and layer scoring."""
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_layer_modules(model, layer_indices: List[int]) -> List[nn.Module]:
    """Return transformer layer modules for the given indices.

    Automatically unwraps Accelerator / DDP / PEFT wrappers.
    """
    raw = _unwrap_model(model)
    pattern = re.compile(r"(?:.*\.)?language_model\.layers\.(\d+)$")
    name2mod = {}
    for name, mod in raw.named_modules():
        m = pattern.fullmatch(name)
        if m:
            name2mod[int(m.group(1))] = mod
    modules = []
    for idx in layer_indices:
        if idx not in name2mod:
            raise ValueError(f"Layer {idx} not found. Available: {sorted(name2mod.keys())}")
        modules.append(name2mod[idx])
    return modules


def forward_with_cache_multilayer(
    model, inputs: Dict, modules: List[nn.Module], no_grad: bool = True
) -> Tuple[List[torch.Tensor], object]:
    """Run one forward pass while caching activations of multiple layers."""
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


class RandomTargetCache:
    """Cache activations of every target layer obtained by passing a random
    token sequence through the reference model."""
    def __init__(self, ref_model, ref_modules: List[nn.Module], vocab_size: int,
                 seq_len: int, device: str, dtype: torch.dtype):
        self.targets = self._build(ref_model, ref_modules, vocab_size, seq_len, device, dtype)

    @torch.no_grad()
    def _build(self, ref_model, ref_modules, vocab_size, seq_len, device, dtype):
        random_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones_like(random_ids)
        inputs = {"input_ids": random_ids, "attention_mask": attention_mask}
        activations, _ = forward_with_cache_multilayer(ref_model, inputs, ref_modules, no_grad=True)
        return [act.detach() for act in activations]

    def get_target(self, layer_idx: int, batch_size: int, target_seq_len: int):
        t = self.targets[layer_idx]
        cache_seq_len = t.shape[1]
        t = t.expand(batch_size, -1, -1)
        if target_seq_len <= cache_seq_len:
            return t[:, :target_seq_len, :]
        else:
            repeats = (target_seq_len // cache_seq_len) + 1
            t = t.repeat(1, repeats, 1)
            return t[:, :target_seq_len, :]


def compute_activation_loss(
    activation1: torch.Tensor, activation2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-token MSE loss with a position mask."""
    squared_diff = nn.functional.mse_loss(activation1, activation2, reduction="none")
    expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
    per_sample = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
    num_tokens = mask.sum(dim=-1).clamp(min=1)
    return (per_sample / num_tokens).mean()


def build_answer_mask_for_multimodal(
    input_ids: torch.Tensor, labels: torch.Tensor, act_len: int, device: torch.device
) -> torch.Tensor:
    """Build an answer-position mask aligned with the activation sequence length
    for the multimodal setting."""
    image_token_ids = [32000, 151655, 151652, 151653]
    is_image = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in image_token_ids:
        is_image = is_image | (input_ids == tid)
    answer_mask = (labels != -100) & ~is_image
    text_len = answer_mask.shape[1]
    if text_len == act_len:
        return answer_mask.to(device)
    elif text_len > act_len:
        return answer_mask[:, :act_len].to(device)
    else:
        mask = torch.zeros(answer_mask.shape[0], act_len, dtype=torch.bool, device=device)
        mask[:, :text_len] = answer_mask
        return mask


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensors in a batch dictionary to the given device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
