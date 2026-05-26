"""
统一数据集类和数据整理函数, 用于多模态大语言模型的微调训练
支持 UMU 和 CLEAR 两种数据集格式

数据集格式说明:
- UMU: JSON 格式, 包含 MM_QA (多模态) 和 UM_QA (单模态) 字段
- CLEAR: JSON 格式, type="image" 为多模态, type="text" 为单模态 (TOFU)
"""

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

image_caption_questions = [
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]

def detect_dataset_type(data_dir: str) -> str:
    """
    根据路径名称检测数据集类型
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        'umu' 或 'clear'
    """
    # 按路径段判断，避免把仓库名里的 UMU-bench 误识别成 UMU 数据集
    path_parts = [part.upper() for part in os.path.normpath(data_dir).split(os.sep)]
    if 'CLEAR' in path_parts:
        return 'clear'
    if 'UMU' in path_parts:
        return 'umu'
    
    raise ValueError(f"Cannot detect dataset type from path: {data_dir}. Path should contain 'UMU' or 'CLEAR'.")


def load_umu_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 UMU 数据集
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        (multimodal_samples, unimodal_samples) 元组
    """
    data_path = os.path.join(data_dir, 'data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    multimodal_samples = []
    unimodal_samples = []
    
    for item in data:
        image_path = item.get('image_path')
        if image_path:
            image_path = os.path.join(data_dir, image_path)
        
        # 处理多模态问答 (MM_QA)
        mm_qa = item.get('MM_QA', {})
        if mm_qa and image_path:
            questions = mm_qa.get('question', {})
            answers = mm_qa.get('answer', {})
            for key in questions.keys():
                if key in answers:
                    multimodal_samples.append({
                        'image_path': image_path,
                        'question': questions[key],
                        'ground_truth': answers[key]
                    })
        
        # 处理单模态问答 (UM_QA)
        um_qa = item.get('UM_QA', {})
        if um_qa:
            questions = um_qa.get('question', {})
            answers = um_qa.get('answer', {})
            for key in questions.keys():
                if key in answers:
                    unimodal_samples.append({
                        'image_path': None,
                        'question': questions[key],
                        'ground_truth': answers[key]
                    })
    
    return multimodal_samples, unimodal_samples


def load_clear_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 CLEAR 数据集
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        (multimodal_samples, unimodal_samples) 元组
    """
    data_path = os.path.join(data_dir, 'data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    multimodal_samples = []
    unimodal_samples = []
    
    for item in data:
        item_type = item.get('type', '')
        
        if item_type == 'image':
            # 多模态: 图像描述任务
            image_path = item.get('image_path')
            caption = item.get('caption')
            question = random.choice(image_caption_questions)
            if image_path and caption:
                image_path = os.path.join(data_dir, image_path)
                multimodal_samples.append({
                    'image_path': image_path,
                    'question': question,
                    'ground_truth': caption
                })
        
        elif item_type == 'text':
            # 单模态: TOFU 文本问答
            question = item.get('question')
            answer = item.get('answer')
            if question and answer:
                unimodal_samples.append({
                    'image_path': None,
                    'question': question,
                    'ground_truth': answer
                })
    
    return multimodal_samples, unimodal_samples


class UnifiedDataset(Dataset):
    """
    统一数据集类, 支持 UMU 和 CLEAR 两种格式
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'multimodal',
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            data_dir: 数据目录路径 (包含 data.json 和 images/ 文件夹)
            mode: 'multimodal' 或 'unimodal'
            target_size: 图像目标尺寸, None 表示保持原尺寸
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        
        # 检测数据集类型
        self.dataset_type = detect_dataset_type(data_dir)
        print(f"Detected dataset type: {self.dataset_type}")
        
        # 加载数据
        if self.dataset_type == 'umu':
            multimodal, unimodal = load_umu_data(data_dir)
        else:  # clear
            multimodal, unimodal = load_clear_data(data_dir)
        
        # 根据模式选择数据
        if mode == 'multimodal':
            self.samples = multimodal
        elif mode == 'unimodal':
            self.samples = unimodal
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'multimodal' or 'unimodal'")
        
        print(f"Loaded {len(self.samples)} {mode} samples from {self.dataset_type} dataset")
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载并处理图像"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.target_size:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        image = None
        if sample.get('image_path'):
            image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'question': sample['question'],
            'ground_truth': sample['ground_truth']
        }


def collate_fn_multimodal(examples: List[Dict], processor, model_type: str = 'llava'):
    """多模态 collate 函数"""
    valid_examples = [ex for ex in examples if ex.get('image') is not None]
    if not valid_examples:
        raise ValueError("Batch contains no valid images!")

    images = [ex['image'] for ex in valid_examples]
    questions = [ex['question'] for ex in valid_examples]
    ground_truths = [ex['ground_truth'] for ex in valid_examples]

    is_qwen = 'qwen' in model_type.lower()
    texts_full, texts_prompt = [], []

    if is_qwen:
        for q, a in zip(questions, ground_truths):
            texts_full.append(processor.apply_chat_template([
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                {"role": "assistant", "content": a},
            ], tokenize=False, add_generation_prompt=False))
            texts_prompt.append(processor.apply_chat_template([
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            ], tokenize=False, add_generation_prompt=True))
    else:
        for q, a in zip(questions, ground_truths):
            texts_full.append(f"USER: <image>\n{q}\nASSISTANT: {a}")
            texts_prompt.append(f"USER: <image>\n{q}\nASSISTANT:")

    batch_prompt = processor(text=texts_prompt, images=images, padding=True, return_tensors="pt")
    batch_full = processor(text=texts_full, images=images, padding=True, return_tensors="pt")

    prompt_lengths = batch_prompt.attention_mask.sum(dim=1)
    full_lengths = batch_full.attention_mask.sum(dim=1)

    input_ids = batch_full["input_ids"]
    labels = input_ids.clone()
    batch_seq_len = input_ids.shape[1]

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    is_left_pad = getattr(processor.tokenizer, 'padding_side', 'right') == 'left'

    for i in range(len(images)):
        p_len = min(prompt_lengths[i].item(), full_lengths[i].item())
        f_len = full_lengths[i].item()

        if is_left_pad:
            mask_end = (batch_seq_len - f_len) + p_len
        else:
            mask_end = p_len

        labels[i, :min(mask_end, batch_seq_len)] = -100

    batch = {
        "input_ids": batch_full["input_ids"],
        "attention_mask": batch_full["attention_mask"],
        "pixel_values": batch_full["pixel_values"],
        "labels": labels,
        "question": questions,
        "ground_truth": ground_truths,
        "inputs": {
            "input_ids": batch_prompt["input_ids"],
            "attention_mask": batch_prompt["attention_mask"],
            "pixel_values": batch_prompt["pixel_values"],
        },
    }
    if is_qwen and "image_grid_thw" in batch_full:
        batch["image_grid_thw"] = batch_full["image_grid_thw"]
    if is_qwen and "image_grid_thw" in batch_prompt:
        batch["inputs"]["image_grid_thw"] = batch_prompt["image_grid_thw"]
    return batch

def collate_fn_unimodal(examples: List[Dict], processor, model_type: str = 'llava'):
    """单模态 collate 函数"""
    questions = [ex['question'] for ex in examples]
    ground_truths = [ex['ground_truth'] for ex in examples]

    is_qwen = 'qwen' in model_type.lower()
    texts_full, texts_prompt = [], []

    if is_qwen:
        for q, a in zip(questions, ground_truths):
            texts_full.append(processor.apply_chat_template([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ], tokenize=False, add_generation_prompt=False))
            texts_prompt.append(processor.apply_chat_template([
                {"role": "user", "content": q},
            ], tokenize=False, add_generation_prompt=True))
    else:
        for q, a in zip(questions, ground_truths):
            texts_full.append(f"USER: {q}\nASSISTANT: {a}")
            texts_prompt.append(f"USER: {q}\nASSISTANT:")

    batch = processor.tokenizer(texts_full, padding=True, return_tensors="pt")
    batch_prompt = processor.tokenizer(texts_prompt, padding=True, return_tensors="pt")

    prompt_lengths = batch_prompt.attention_mask.sum(dim=1)
    full_lengths = batch.attention_mask.sum(dim=1)

    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    batch_seq_len = input_ids.shape[1]

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    is_left_pad = getattr(processor.tokenizer, 'padding_side', 'right') == 'left'

    for i in range(len(questions)):
        p_len = min(prompt_lengths[i].item(), full_lengths[i].item())
        f_len = full_lengths[i].item()

        if is_left_pad:
            mask_end = (batch_seq_len - f_len) + p_len
        else:
            mask_end = p_len

        labels[i, :mask_end] = -100

    # 返回 dict，单模态无图像故 pixel_values 为 None；inputs 供 model.generate(**inputs) 用，与多模态一致
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": None,
        "labels": labels,
        "question": questions,
        "ground_truth": ground_truths,
        "inputs": {
            "input_ids": batch_prompt["input_ids"],
            "attention_mask": batch_prompt["attention_mask"],
            "pixel_values": None,
        },
    }