"""
微调模块 - 用于多模态大语言模型的微调训练

支持的模型:
- LLaVA-1.5-7B
- Qwen2.5-VL-3B

支持的数据集格式:
- UMU: JSON 格式, 包含 MM_QA (多模态) 和 UM_QA (单模态) 字段
- CLEAR: JSON 格式, type="image" 为多模态, type="text" 为单模态 (TOFU)
"""

from finetune.dataset import (
    UnifiedDataset,
    collate_fn_multimodal,
    collate_fn_unimodal,
    detect_dataset_type,
    load_umu_data,
    load_clear_data,
)

__all__ = [
    'UnifiedDataset',
    'MultimodalDataset',
    'UnimodalDataset',
    'collate_fn_multimodal',
    'collate_fn_unimodal',
    'detect_dataset_type',
    'load_umu_data',
    'load_clear_data',
]
