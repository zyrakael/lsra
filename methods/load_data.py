# load_data.py
import os
from typing import Tuple, List, Optional
from torch.utils.data import DataLoader

from finetune.dataset import UnifiedDataset
from finetune.dataset import (
    collate_fn_multimodal,
    collate_fn_unimodal
)

from utils.model_utils import MODELS

def _build_dataloader(
    data_dir: str,
    mode: str,
    processor,
    args,
    shuffle: bool = True,
):
    dataset = UnifiedDataset(
        data_dir=data_dir,
        mode=mode,
        target_size=(args.image_resize, args.image_resize) if mode == "multimodal" else None
    )
    model_type = MODELS[args.model]['type']
    if mode == "multimodal":
        collate_fn = lambda x: collate_fn_multimodal(x, processor, model_type)
    else:
        collate_fn = lambda x: collate_fn_unimodal(x, processor, model_type)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return loader



def resolve_forget_retain_dirs(args):
    """
    Resolve the forget / retain data directories from args.
    """
    dataset = args.dataset.lower()
    base_dir = args.data_dir
    forget_ratio = args.forget_ratio
    retain_ratio = 100 - forget_ratio

    if dataset == "umu":
        forget_dir = os.path.join(
            base_dir, "UMU", f"forget_{forget_ratio}"
        )
        retain_dir = os.path.join(
            base_dir, "UMU", f"retain_{retain_ratio}"
        )

    elif dataset == "clear":
        forget_dir = os.path.join(
            base_dir, "CLEAR", f"forget{forget_ratio:02d}_plus_tofu"
        )
        retain_dir = os.path.join(
            base_dir, "CLEAR", f"retain{retain_ratio}_plus_tofu"
        )

    return forget_dir, retain_dir

def load_forget_dataloaders(
    processor,
    args,
) -> Tuple[DataLoader, DataLoader]:
    forget_data_dir, _ = resolve_forget_retain_dirs(args)

    loader_multi_forget = _build_dataloader(
        data_dir=forget_data_dir,
        mode="multimodal",
        processor=processor,
        args=args,
        shuffle=False
    )

    loader_uni_forget = _build_dataloader(
        data_dir=forget_data_dir,
        mode="unimodal",
        processor=processor,
        args=args,
        shuffle=False
    )

    return loader_multi_forget, loader_uni_forget


def load_retain_dataloaders(
    processor,
    args,
) -> Tuple[DataLoader, DataLoader]:
    _, retain_data_dir = resolve_forget_retain_dirs(args)

    loader_multi_retain = _build_dataloader(
        data_dir=retain_data_dir,
        mode="multimodal",
        processor=processor,
        args=args,
        shuffle=True,
    )

    loader_uni_retain = _build_dataloader(
        data_dir=retain_data_dir,
        mode="unimodal",
        processor=processor,
        args=args,
        shuffle=True,
    )

    return loader_multi_retain, loader_uni_retain

