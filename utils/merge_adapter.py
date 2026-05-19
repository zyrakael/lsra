"""
Merge a LoRA adapter into the base model and export the full model.

Usage:
    python utils/merge_adapter.py \
        --adapter_path outputs/llava_umu_llamafactory \
        --output_path weight/LLaVA-1.5-7B-merged

    # Manually specify the base model path (overrides base_model_name_or_path in adapter_config.json)
    python utils/merge_adapter.py \
        --adapter_path weight/GD_LLaVA-1.5-7B_umu_20260515_162953 \
        --base_model_path weight/LLaVA-1.5-7B-UMU\
        --output_path weight/LLaVA-1.5-7B-GD_15
"""

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CLASS_MAP = {
    "llava": LlavaForConditionalGeneration,
    "qwen": Qwen2_5_VLForConditionalGeneration,
}


def detect_model_type(base_path: str) -> str:
    config_path = os.path.join(base_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    arch = config.get("architectures", [""])[0].lower()
    if "llava" in arch:
        return "llava"
    if "qwen" in arch:
        return "qwen"
    raise ValueError(f"Unrecognized model type, architectures={config.get('architectures')}")


def resolve_base_path(adapter_path: str, base_model_path: str | None) -> str:
    if base_model_path:
        p = base_model_path
    else:
        adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
        with open(adapter_cfg_path, "r") as f:
            cfg = json.load(f)
        p = cfg.get("base_model_name_or_path", "")
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p)
    return p


def merge(adapter_path: str, base_model_path: str | None, output_path: str):
    base_path = resolve_base_path(adapter_path, base_model_path)
    print(f"Base model path: {base_path}")
    print(f"Adapter path:    {adapter_path}")
    print(f"Output path:     {output_path}")

    model_type = detect_model_type(base_path)
    model_cls = MODEL_CLASS_MAP[model_type]
    print(f"Model type: {model_type} ({model_cls.__name__})")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base model (dtype={dtype}) ...")
    base_model = model_cls.from_pretrained(
        base_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="cpu",
    )

    print("Loading LoRA adapter ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights ...")
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    print("Copying processor / tokenizer files ...")
    processor = AutoProcessor.from_pretrained(adapter_path, local_files_only=True)
    processor.save_pretrained(output_path)

    print("Done. The merged full model has been saved.")


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/llava_umu_llamafactory_vision_false",
        help="LoRA adapter directory (must contain adapter_config.json)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default='weight/LLaVA-1.5-7B',
        help="Base model path (optional, defaults to value read from adapter_config.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='weight/LLaVA-1.5-7B-vision-false',
        help="Output directory for the merged full model",
    )
    args = parser.parse_args()

    adapter_path = args.adapter_path
    if not os.path.isabs(adapter_path):
        adapter_path = os.path.join(PROJECT_ROOT, adapter_path)

    base_model_path = args.base_model_path
    if base_model_path and not os.path.isabs(base_model_path):
        base_model_path = os.path.join(PROJECT_ROOT, base_model_path)

    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    merge(adapter_path, base_model_path, output_path)


if __name__ == "__main__":
    main()
