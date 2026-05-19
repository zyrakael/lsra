"""
Model loading and forward-pass utilities for multimodal LLMs.
"""

import json
import math
import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Supported model configurations.
MODELS = {
    'LLaVA-1.5-7B': {
        'path': os.path.join(PROJECT_ROOT, 'weight', 'LLaVA-1.5-7B-UMU'),
        'processor_id': 'llava-hf/llava-1.5-7b-hf',
        'model_class': LlavaForConditionalGeneration,
        'type': 'llava',
    },
    'Qwen2.5-VL-3B': {
        'path': os.path.join(PROJECT_ROOT, 'weight', 'Qwen2.5-VL-3B-UMU'),
        'processor_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'model_class': Qwen2_5_VLForConditionalGeneration,
        'type': 'qwen',
    },
}


def get_supported_models() -> list:
    """Return the list of supported model names."""
    return list(MODELS.keys())


def load_model_and_processor(
    model_name: str,
    model_path: str = None,
    model_device: str = None,
    is_trainable: bool = True,
):
    """
    Load a model and its processor.

    When the given path is a LoRA checkpoint (contains ``adapter_config.json``),
    the base model and the adapter are loaded jointly, so that training can
    continue on top of an existing LoRA and evaluation runs on the fine-tuned
    model.
    """
    config = MODELS[model_name]
    load_path = model_path if model_path else config['path']
    adapter_config_path = os.path.join(load_path, "adapter_config.json")
    is_peft_checkpoint = os.path.exists(adapter_config_path)

    print(f"[load_model] load_path={load_path}")
    print(f"[load_model] adapter_config_path={adapter_config_path}")
    print(f"[load_model] is_peft_checkpoint={is_peft_checkpoint}")

    if is_peft_checkpoint:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_path = adapter_config.get('base_model_name_or_path')
    else:
        base_path = load_path

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = {"": model_device} if model_device is not None else None

    if is_peft_checkpoint:
        # PEFT 0.15.2 fails to load adapter weights correctly when device_map
        # points directly to a GPU. Load the base model + LoRA adapter on CPU
        # first, then move the whole thing to the target device.
        model = config['model_class'].from_pretrained(
            base_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map="cpu",
        )
        print(f"[load_model] Loading LoRA adapter from {load_path} ...")
        model = PeftModel.from_pretrained(model, load_path, is_trainable=is_trainable)
        # Sanity-check that the LoRA weights were loaded correctly.
        for name, param in model.named_parameters():
            if 'layers.0.self_attn.q_proj.lora_B' in name:
                val = param.data.float().abs().mean().item()
                print(f"[load_model] LoRA B check: {name} abs_mean={val:.8f} {'OK' if val > 0 else 'ZERO - NOT LOADED!'}")
                break
        if not is_trainable:
            model = model.merge_and_unload()
            print(f"[load_model] LoRA merged into base model")
        if model_device is not None:
            model = model.to(model_device)
            print(f"[load_model] Model moved to {model_device}")
    else:
        model = config['model_class'].from_pretrained(
            base_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=device_map,
        )
    processor_path = load_path if is_peft_checkpoint else base_path
    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
        use_fast=True,
        fix_mistral_regex=True,
    )

    if config['type'] == 'llava':
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif config['type'] == 'qwen':
        processor.tokenizer.padding_side = "right"

    return model, processor


def forward_pass(batch, model, model_name: str, mode: str):
    """
    Forward pass.

    Args:
        batch: A batch dict (input_ids, attention_mask, pixel_values, labels,
            optionally image_grid_thw).
        model: Model instance.
        model_name: Model name.
        mode: 'multimodal' or 'unimodal'.

    Returns:
        Model outputs.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")

    if mode == 'multimodal':
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        outputs = model(**kwargs)
    else:  
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    return outputs


def get_label_token_probs(outputs, labels):
    """
    Compute the (log-)probability of each label token from ``outputs.logits``.

    For a causal LM, ``logits[b, t, :]`` predicts the next token at position
    ``t+1``, so the target is ``labels[b, t+1]``. We align with
    ``logits[:, :-1, :]`` and ``labels[:, 1:]``.

    Args:
        outputs: Model output with ``.logits`` of shape (batch, seq_len, vocab_size).
        labels: (batch, seq_len). Positions with -100 are ignored.

    Returns:
        token_log_probs: (batch, seq_len - 1) log-probability of the gold token
            at each valid position. Positions with label == -100 are -inf.
        valid_mask: bool mask of valid positions.
    """
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    shift_log_probs = F.log_softmax(shift_logits, dim=-1)

    valid_mask = (shift_labels != -100)
    
    # Replace -100 with 0 in a copy so that ``gather`` does not index OOB.
    target_labels = shift_labels.clone()
    target_labels[target_labels == -100] = 0
    
    token_log_probs = torch.gather(shift_log_probs, dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs, valid_mask


def get_predictive_entropy(outputs, labels=None):
    """
    Compute the predictive entropy and return the *confidence* ``1 - H_norm``,
    where ``H = -sum_i P_i log P_i`` is summed over the vocabulary and
    ``H_norm = H / log V``.

    Args:
        outputs: Model output with ``.logits`` of shape (batch, seq_len, vocab_size).
        labels: Optional (batch, seq_len). When provided, only answer positions
            (labels != -100) are averaged, returning one scalar per sample.

    Returns:
        If labels is None: (batch, seq_len - 1) of ``1 - H_norm`` per position.
        Otherwise: (batch,) average ``1 - H_norm`` over answer tokens.
    """
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    vocab_size = shift_logits.shape[-1]
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    log_V = math.log(vocab_size)
    conf = 1 - (entropy / log_V)

    if labels is not None:
        shift_labels = labels[..., 1:].contiguous()
        valid = (shift_labels != -100).float()
        n = valid.sum(dim=1).clamp(min=1)
        conf = (conf * valid).sum(dim=1) / n
    return conf


def get_valid_position_counts(labels):
    """
    Number of valid (!= -100) positions per sample after the causal shift, i.e.
    aligned with ``labels[:, 1:]``.

    Args:
        labels: [B, L]; -100 marks ignored positions.

    Returns:
        counts: [B]; valid-position count per sample.
    """
    shift_labels = labels[:, 1:]
    valid = (shift_labels != -100)
    return valid.sum(dim=1)


def get_unlearning_weight(logits, labels, beta=1.0, k=20, m=10, n_valid_threshold=200):
    """
    Dynamic weight for gradient ascent during unlearning, scaled by whether the
    model currently answers correctly. Only positions with
    ``shift_labels != -100`` participate. We compute ``s_t = exp(beta * margin)``
    where ``margin = L_target - L_max``, then average ``s_t`` (sampled or full).

    Logic:
        - Causal shift: valid = (shift_labels != -100); margin_t = L_target - L_max; s_t = exp(beta * margin_t).
        - Mask invalid tokens (only positions with valid == True participate).
        - Conditional sampling: if N_valid > n_valid_threshold, run k trials,
          each sampling m valid positions and averaging s_t; then average over
          the k trials. Otherwise just average s_t over all valid positions.

    Args:
        logits: [B, L, V].
        labels: [B, L]; -100 marks ignored positions.
        beta: Temperature.
        k: Number of trials when N_valid > n_valid_threshold (default 20).
        m: Number of valid positions sampled per trial (default 10).
        n_valid_threshold: Switch threshold (default 200).

    Returns:
        weight: [B], one scalar per sample.
        margin_mean: [B], mean margin over valid positions per sample.
    """
    B, L, V = logits.shape
    T = L - 1
    device = logits.device
    dtype = logits.dtype

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = (shift_labels != -100)

    L_max, _ = shift_logits.max(dim=-1)
    target_idx = shift_labels.clone()
    target_idx[target_idx == -100] = 0
    L_target = torch.gather(
        shift_logits, 2, target_idx.unsqueeze(2).clamp(0, V - 1)
    ).squeeze(2)
    margin = L_target - L_max  # [B, T]
    s_t = torch.exp(beta * margin)

    weights = []
    margin_means = []  # Per-sample mean margin over valid positions, for logging.
    for b in range(B):
        valid_b = valid[b]  # [T]
        n_valid = valid_b.sum().item()
        s_t_b = s_t[b]  # [T]
        margin_b = margin[b]  # [T]
        if n_valid == 0:
            weights.append(torch.tensor(0.0, device=device, dtype=dtype))
            margin_means.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue
        margin_means.append(margin_b[valid_b].mean())
        if n_valid > n_valid_threshold:
            # k trials, each sampling m valid positions without replacement;
            # average s_t within each trial, then average across trials.
            valid_indices = valid_b.nonzero().squeeze(-1)  # [n_valid]
            trial_means = []
            for _ in range(k):
                perm = torch.randperm(n_valid, device=device)[:m]
                selected_pos = valid_indices[perm]  # [m]
                trial_means.append(s_t_b[selected_pos].mean())
            weights.append(torch.stack(trial_means).mean())
        else:
            # Average s_t over all valid positions.
            weights.append(s_t_b[valid_b].mean())
    weight_out = weights[0].unsqueeze(0) if B == 1 else torch.stack(weights)
    margin_out = margin_means[0].unsqueeze(0) if B == 1 else torch.stack(margin_means)
    return weight_out, margin_out 


def per_position_kl_divergence(oracle_out, out_uf, labels):
    """
    Per-position KL divergence between the oracle and the model under test on
    answer positions only. The gold-answer token is masked out of both
    distributions so the KL is computed on the remaining vocabulary
    (the "distractors").

    Logic:
        - Causal shift: align logits[:, :-1, :] with labels[:, 1:].
        - Answer mask: mask = (shifted_labels != -100).
        - Mask gold answer: set its logit to -inf in both distributions, so
          softmax assigns it probability 0.
        - KL(P || Q) where P = softmax(oracle masked), Q = softmax(model masked);
          sum over vocab to get [B, S-1], then average over valid positions.

    Args:
        oracle_out: Output with ``.logits`` of shape [B, S, V].
        out_uf: Output with ``.logits`` of shape [B, S, V].
        labels: [B, S]; prompt tokens are -100, answer tokens are gold ids.

    Returns:
        kl_mean: [B], average KL over answer positions per sample.
    """
    logits_o = oracle_out.logits[:, :-1, :].contiguous()
    logits_u = out_uf.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    B, T, V = logits_o.shape
    device = logits_o.device
    dtype = logits_o.dtype

    mask = (shifted_labels != -100)

    idx = shifted_labels.clone()
    idx[idx == -100] = 0
    idx = idx.unsqueeze(2).clamp(0, V - 1)
    neg_inf_src = torch.full((B, T, 1), -1e9, device=device, dtype=dtype)

    logits_o_masked = logits_o.clone()
    logits_u_masked = logits_u.clone()
    logits_o_masked.scatter_(2, idx, neg_inf_src)
    logits_u_masked.scatter_(2, idx, neg_inf_src)

    P = F.softmax(logits_o_masked, dim=-1)
    log_Q = F.log_softmax(logits_u_masked, dim=-1)
    kl_per_pos = F.kl_div(log_Q, P, reduction="none").sum(dim=-1)
    kl_per_pos = kl_per_pos * mask.float()
    n_valid = mask.sum(dim=1).clamp(min=1)
    kl_mean = kl_per_pos.sum(dim=1) / n_valid
    return kl_mean


def find_all_linear_names(model):
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def get_lora_config(model, method: str = 'default') -> LoraConfig:
    """
    Build a LoRA config.

    Args:
        model: Model instance.
        method: Method name.
            - 'default': use ``find_all_linear_names``.
            - 'ga': GA method uses a fixed list of target modules.
            - 'gd': GD method uses only ``q_proj`` and ``v_proj``.

    Returns:
        A ``LoraConfig`` instance.
    """
    if method == 'ga':
        target_modules = ['o_proj', 'q_proj', 'k_proj', 'down_proj', 'v_proj', 'up_proj', 'gate_proj']
    elif method == 'gd':
        target_modules = ['q_proj', 'v_proj']
    else:
        target_modules = find_all_linear_names(model)
    
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model, method: str = 'default'):
    """
    Wrap the model with a LoRA adapter.

    Args:
        model: Model instance.
        method: Method name (see ``get_lora_config``).

    Returns:
        The model with LoRA applied.
    """
    lora_config = get_lora_config(model, method)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
