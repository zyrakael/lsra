"""
多模态大语言模型微调训练脚本, 支持 LLaVA-1.5-7B 和 Qwen2.5-VL-3B, 支持 UMU 和 CLEAR 两种数据集格式
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,5"
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    load_model_and_processor,
    find_all_linear_names,
    get_supported_models,
    MODELS,
)
from finetune.dataset import (
    UnifiedDataset,
    collate_fn_multimodal,
    collate_fn_unimodal,
)

save_dir = '/sda/home/wangmengyao/Mutimodal_Unlearning/UMU-bench/weight'
data_dir = '/sda/home/wangmengyao/Mutimodal_Unlearning/UMU-bench/datasets/UMU/full_set'

def main(args):
    # 加载模型和处理器
    model, processor = load_model_and_processor(args.model, args.model_path)
    model_type = MODELS[args.model]['type']
    
    print(f"Model: {args.model}")
    print(f"Model type: {model_type}")
    print(f"Processor tokenizer length: {len(processor.tokenizer)}")
    
    # 调整词嵌入大小
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    
    # 加载数据集
    print(f"\nLoading dataset from: {args.data_dir}")
    multimodal_dataset = UnifiedDataset(
        data_dir=args.data_dir,
        mode='multimodal'
    )
    unimodal_dataset = UnifiedDataset(
        data_dir=args.data_dir,
        mode='unimodal'
    )
    
    print(f"Multimodal samples: {len(multimodal_dataset)}")
    print(f"Unimodal samples: {len(unimodal_dataset)}")
    
    # 创建数据加载器
    dataloader_multi = DataLoader(
        multimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn_multimodal(x, processor, model_type)
    )
    dataloader_uni = DataLoader(
        unimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn_unimodal(x, processor, model_type)
    )
    
    # 初始化加速器
    accelerator = Accelerator()
    
    # 计算总训练步数
    total_steps = len(dataloader_multi) * args.num_epochs
    if args.train_unimodal:
        total_steps += len(dataloader_uni) * args.num_epochs
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    
    # 准备分布式训练
    model, optimizer, dataloader_multi, dataloader_uni, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader_multi, dataloader_uni, lr_scheduler
    )
    
    # 训练循环
    print(f"\nStarting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        # 多模态训练
        if len(multimodal_dataset) > 0:
            progress_bar = tqdm(
                dataloader_multi,
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [Multimodal]"
            )
            for batch in progress_bar:
                # Qwen2.5-VL 返回 5 个值，LLaVA 返回 4 个值
                if len(batch) == 5:
                    input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels
                    )
                else:
                    input_ids, attention_mask, pixel_values, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels
                    )
                
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader_multi)
            print(f"Epoch {epoch + 1} - Multimodal Average Loss: {avg_loss:.4f}")
        
        # 单模态训练
        if args.train_unimodal and len(unimodal_dataset) > 0:
            total_loss = 0
            progress_bar = tqdm(
                dataloader_uni,
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [Unimodal]"
            )
            for batch in progress_bar:
                input_ids, attention_mask, _, labels = batch
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader_uni)
            print(f"Epoch {epoch + 1} - Unimodal Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    print("\nSaving model...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune multimodal LLM on UMU or CLEAR dataset"
    )
    
    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default='Qwen2.5-VL-3B',
        choices=get_supported_models(),
        help="Model name"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Custom model path (optional)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default= save_dir,
        help="Directory to save the finetuned model"
    )
    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        default = data_dir,
        help="Path to dataset directory (containing data.json and images/)"
    )
    
    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--train_unimodal",
        action="store_true",
        help="Also train on unimodal data"
    )
    
    # LoRA 参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    
    # 其他参数
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    main(args)
