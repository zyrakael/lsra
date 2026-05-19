import sys, os

import time
import json
from datetime import datetime
from sympy import N
import torch

# Add the project root to sys.path so that imports such as utils / eval /
# finetune can be resolved below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed_utils import set_seed

from eval.evaluate import evaluate_clf, evaluate_gen

import torch
from torch.utils.data import DataLoader
import argparse


from eval.load_data import load_retain_dataloaders
from eval.load_data import load_forget_dataloaders
from eval.load_data import build_umu_clf_dataloaders
from peft import PeftModel
from utils.model_utils import (
    load_model_and_processor,
    find_all_linear_names,
    get_supported_models,
    apply_lora,
    MODELS,
)

from finetune.dataset import (
    UnifiedDataset,
    collate_fn_multimodal,
    collate_fn_unimodal,
)

seed = 42
image_resize = 224
forget_ratio =5
batch_size = 1
judge_device = "cuda:5"
model_device = "cuda:5"
base_path = "/sda/home/wangmengyao/Mutimodal_Unlearning/UMU-bench/datasets"
dataset = "clear"
score_llm = "Qwen2-7B-Instruct"
llm_directory = "/sda/home/wangmengyao/Mutimodal_Unlearning/UMU-bench/weight/"
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument(
    "--model",
    type=str,
    default='Qwen2.5-VL-3B',
    # LLaVA-1.5-7B,Qwen2.5-VL-3B
    help="Model name"
)
parser.add_argument(
    "--model_path",
    type=str,
    default='outputs/multilayer_rmu/Qwen2.5-VL-3B_clear_epoch9_step846_20260407_121018',
    help="Custom model path (optional)"
)
# Data arguments
parser.add_argument(
    "--base_path", 
    type=str, 
    default=base_path
)
parser.add_argument(
    "--dataset", 
    type=str, 
    default=dataset
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=batch_size
)
# Evaluation arguments
parser.add_argument(
    "--image_resize", 
    type=int, 
    default=image_resize
)
parser.add_argument(
    "--forget_ratio", 
    type=int, 
    default=forget_ratio
)
parser.add_argument(
    "--score_llm", 
    type=str, 
    default=score_llm
)
parser.add_argument(
    "--judge_device", 
    type=str, 
    default=judge_device,
    help="Device for judge model (e.g., cuda:0)"
)
parser.add_argument(
    "--model_device", 
    type=str, 
    default=model_device,
    help="Device for main model (e.g., cuda:1)"
)
parser.add_argument(
    "--device", 
    type=str, 
    default=model_device,
    help="Legacy device parameter (defaults to model_device)"
)
parser.add_argument(
    "--llm_directory",
    type=str,
    default=llm_directory,
    help="Directory for judge LLM models"
)
parser.add_argument(
    "--this_run_id",
    type=str,
    default=time.strftime("%m%d_%H%M%S"),
    help="Unique identifier for this run (default: timestamp)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=seed,
    help="Random seed"
)
parser.add_argument(
    "--output_file_path",
    type=str,
    default="./eval_results",
    help="Directory to save evaluation results"
)
parser.add_argument(
    "--replace_vision_encoder",
    action="store_true",
    default=False,
    help="Whether to replace the vision components (encoder + projector)."
)
parser.add_argument(
    "--original_vision_weight_dir",
    type=str,
    default='weight/LLaVA-1.5-7B',
    help="Directory containing the original vision-component weights "
         "(must contain safetensors files). Defaults to weight/<model_name>."
)

args = parser.parse_args()


def replace_vision_components(model, model_name: str, device: str, original_weight_dir: str):
    """
    Load the original vision encoder and projector weights from
    ``original_weight_dir`` and replace the corresponding parts of the current
    model (vision_tower / multi_modal_projector / visual.*).

    Prefix rules:
      - LLaVA family: vision_tower.* + multi_modal_projector.*
      - Qwen family:  visual.* (encoder blocks / merger / patch_embed)
    """
    from safetensors import safe_open
    import glob

    assert os.path.isdir(original_weight_dir), \
        f"Original weight directory does not exist: {original_weight_dir}"

    model_type = MODELS[model_name].get('type', '')
    if model_type == 'llava':
        prefixes = ('vision_tower.', 'multi_modal_projector.')
    elif model_type == 'qwen':
        prefixes = ('visual.',)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}; cannot determine vision-component prefixes."
        )

    safetensor_files = sorted(glob.glob(os.path.join(original_weight_dir, "model*.safetensors")))
    assert safetensor_files, f"No safetensors weight files found under {original_weight_dir}"

    vision_state_dict = {}
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.startswith(p) for p in prefixes):
                    vision_state_dict[key] = f.get_tensor(key)

    assert vision_state_dict, f"No vision-component parameters found under prefixes {prefixes}"

    target_model = model.base_model.model if isinstance(model, PeftModel) else model
    current_sd = target_model.state_dict()
    sample_key = list(vision_state_dict.keys())[0]
    needs_prefix = (sample_key not in current_sd) and (f"model.{sample_key}" in current_sd)

    replaced, skipped = 0, 0
    for key, value in vision_state_dict.items():
        mapped_key = f"model.{key}" if needs_prefix else key
        if mapped_key in current_sd:
            current_sd[mapped_key] = value.to(current_sd[mapped_key].dtype)
            replaced += 1
        else:
            skipped += 1

    target_model.load_state_dict(current_sd, strict=False)
    print(f"\n[OK] Vision components replaced ({prefixes}): {replaced} parameters loaded from {original_weight_dir}")
    if skipped:
        print(f"[WARN] Skipped {skipped} parameters that do not exist in the current model")


def main():
    set_seed(args.seed)
    model, processor = load_model_and_processor(
        args.model,
        args.model_path,
        model_device=args.model_device,
        is_trainable=False,
    )

    if args.replace_vision_encoder:
        weight_dir = args.original_vision_weight_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "weight", args.model
        )
        replace_vision_components(model, args.model, args.model_device, weight_dir)

    # Build dataloaders.
    forget_clf_loaders, forget_gen_loaders = load_forget_dataloaders( 
        processor, args
    )
    retain_clf_loaders, retain_gen_loaders = load_retain_dataloaders(
        processor, args
    )

    FVQA, _ = evaluate_clf(forget_clf_loaders["multi"], processor, "forget", "multi", args, model)
    
    # RVQA, _ = evaluate_clf(retain_clf_loaders["multi"], processor, "retain", "multi", args, model)
    
    FVGEN, _ = evaluate_gen(forget_gen_loaders["multi"], processor, "forget", "multi", args, model)
    # # # results['FVGEN'] = FVGEN
    # # # print(f"FVGEN (Forget VQA Gen): {FVGEN:.2f}")
    
    # RVGEN, _ = evaluate_gen(retain_gen_loaders["multi"], processor, "retain", "multi", args, model)
    # # results['RVGEN'] = RVGEN
    # # print(f"RVGEN (Retain VQA Gen): {RVGEN:.2f}")
    
    if args.dataset == "umu":
        FQA, _ = evaluate_clf(forget_clf_loaders["text"], processor, "forget", "text", args, model)
        # RQA, log_dir = evaluate_clf(retain_clf_loaders["text"], processor, "retain", "text", args, model)
    elif args.dataset == "clear":
        FGEN, _ = evaluate_gen(forget_gen_loaders["text"], processor, "forget", "text", args, model)
        # RGEN, log_dir = evaluate_gen(retain_gen_loaders["text"], processor, "retain", "text", args, model)
    # H-Mean = 6 / (1/|0.76-FVQA| + 1/RVQA + 1/|0.99-FVGEN| + 1/RVGEN + 1/|0.82-FQA| + 1/RQA)
    # h_mean = 6.0 / (
    #     1.0 / abs(0.72 - 0) +
    #     1.0 / 0.6357 +
    #     1.0 / abs(0.6672 - 0) +
    #     1.0 / 0.6065 +
    #     1.0 / abs(0.76 - 0.02) +
    #     1.0 / 0.74
    # )
    # print(f"\nH-Mean: {h_mean:.4f}")

    # # Save H-Mean summary to the same directory as evaluate results
    # summary = {
    #     "FVQA": FVQA,
    #     "RVQA": RVQA,
    #     "FVGEN": FVGEN,
    #     "RVGEN": RVGEN,
    #     "FQA": FQA,
    #     "RQA": RQA,
    #     "H-Mean": h_mean
    # }
    # summary_path = os.path.join(log_dir, "hmean_summary.json")
    # with open(summary_path, "w") as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)

    # # print(f"FGEN (Forget QA Gen): {FGEN:.2f}")
    # print(f"RGEN (Retain QA Gen): {RGEN:.2f}")
    # H-Mean = 6 / (1/|0.76-FVQA| + 1/RVQA + 1/|0.99-FVGEN| + 1/RVGEN + 1/|0.82-FQA| + 1/RQA)
    # h_mean = 6.0 / (
    #     1.0 / abs(0.5106 - FVQA) +
    #     1.0 / RVQA +
    #     1.0 / abs(0.5166 - FVGEN) +
    #     1.0 / RVGEN +
    #     1.0 / abs(0.4767 - FGEN) +
    #     1.0 / RGEN
    # )
    # print(f"\nH-Mean: {h_mean:.4f}")

    # Save H-Mean summary to the same directory as evaluate results
    # summary = {
    #     "FVQA": FVQA,
    #     "RVQA": RVQA,
    #     "FVGEN": FVGEN,
    #     "RVGEN": RVGEN,
    #     "FGEN": FGEN,
    #     "RGEN": RGEN,
    #     "H-Mean": h_mean
    # }
    # summary_path = os.path.join(log_dir, "hmean_summary.json")
    # with open(summary_path, "w") as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)
    # Print the final result summary
    # print("\n" + "="*80)
    # print("📊 Evaluation result summary")
    # print("="*80)
    # if SVQA is not None:
    #     print(f"\n🟡 Surrogate Set:")
    #     print(f"  - SVQA (multimodal VQA accuracy): {SVQA:.2f}%")
    # # print(f"\n🔴 Forget Set:")
    # print(f"  - FVQA (multimodal VQA accuracy): {FVQA:.2f}%")
    # # print(f"  - FQA  (unimodal QA accuracy):  {FQA:.2f}%")
    # # print(f"\n🟢 Retain Set:")
    # print(f"  - RVQA (multimodal VQA accuracy): {RVQA:.2f}%")
    # # print(f"  - RQA  (unimodal QA accuracy):  {RQA:.2f}%")
    # print("="*80 + "\n")

if __name__ == "__main__":
    main()
