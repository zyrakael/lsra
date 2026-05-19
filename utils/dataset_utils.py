"""
Dataset classes and collate functions for multimodal / unimodal unlearning.
"""

import json
import ast
import random
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    Multimodal dataset: image + text question-answer pairs.

    Flattens the raw dataframe into individual (image, question, answer) samples.
    """
    
    def __init__(self, df: pd.DataFrame, mode='forget_5', target_size=None):
        """
        Args:
            df: DataFrame holding the raw data.
            mode: Data mode, e.g. 'forget_5' or 'retain_95'.
            target_size: Target image size; None keeps the original size.
        """
        self.df = df
        self.mode = mode
        self.target_size = target_size
        self.dataset = self._flatten_dataset()
    
    def _flatten_dataset(self):
        """Flatten the dataframe into individual QA pairs."""
        flattened = []
        
        for idx, row in self.df.iterrows():
            image_data = row['image'].get('bytes')
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Failed to load image (index {idx}): {e}")
                continue
            
            qa_dict = ast.literal_eval(row['MM_QA'])
            qa_data = json.loads(json.dumps(qa_dict))
            questions = qa_data['question']
            answers = qa_data['answer']
            
            for key in questions.keys():
                flattened.append({
                    "image": image,
                    "question": questions[key],
                    "answer": answers[key]
                })
        
        # Retain-set sampling (balance the forget / retain ratio).
        if self.mode.startswith('retain'):
            ratio = int(self.mode.split('_')[1]) / 100
            n = int(len(flattened) * (1 - ratio) / ratio)
            random.seed(42)
            flattened = random.sample(flattened, min(n, len(flattened)))
        
        return flattened
    
    def _resize_image(self, image):
        """Resize the image to the configured target size."""
        if self.target_size:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "image": self._resize_image(sample["image"]),
            "question": sample["question"],
            "answer": sample["answer"]
        }


class UnimodalDataset(Dataset):
    """
    Unimodal dataset: text-only question-answer pairs.
    """
    
    def __init__(self, df: pd.DataFrame, mode='forget_5'):
        """
        Args:
            df: DataFrame holding the raw data.
            mode: Data mode, e.g. 'forget_5' or 'retain_95'.
        """
        self.df = df
        self.mode = mode
        self.dataset = self._flatten_dataset()
    
    def _flatten_dataset(self):
        """Flatten the dataframe into individual QA pairs."""
        flattened = []
        
        for idx, row in self.df.iterrows():
            qa_dict = ast.literal_eval(row['UM_QA'])
            qa_data = json.loads(json.dumps(qa_dict))
            questions = qa_data['question']
            answers = qa_data['answer']
            
            for key in questions.keys():
                flattened.append({
                    "image": None,
                    "question": questions[key],
                    "answer": answers[key]
                })
        
        # Retain-set sampling.
        if self.mode.startswith('retain'):
            ratio = int(self.mode.split('_')[1]) / 100
            n = int(len(flattened) * (1 - ratio) / ratio)
            random.seed(42)
            flattened = random.sample(flattened, min(n, len(flattened)))
        
        return flattened
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "image": None,
            "question": sample["question"],
            "answer": sample["answer"]
        }


def collate_fn_multimodal(examples, processor, args):
    """
    Multimodal collate function: turns a list of samples into model inputs.
    """
    images, texts = [], []
    
    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        images.append(image)
        texts.append(f"USER: <image>\n{question}\nASSISTANT: {answer}")
    
    if not texts or not images:
        raise ValueError("Empty batch: no valid images or text")
    
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Set labels; pad-token positions are masked with -100 (ignored in loss).
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def collate_fn_unimodal(examples, processor, args):
    """
    Unimodal collate function: turns a list of samples into model inputs (no images).
    """
    texts = []
    
    for example in examples:
        question = example.get('question')
        answer = example.get('answer')
        texts.append(f"USER: {question}\nASSISTANT: {answer}")
    
    if not texts:
        raise ValueError("Empty batch: no valid text")
    
    batch = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch["input_ids"], batch["attention_mask"], None, batch["labels"]
