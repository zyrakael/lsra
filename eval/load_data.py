import torch
from eval.test_data import ClearClfDataset, ClearGenDataset, UMUClfDataset, UMUGenDataset, collator


def build_umu_clf_dataloaders(
    data_path,
    processor,
    args,
    batch_size,
    shuffle: bool = False,
):
    """
    Build UMU classification dataloaders for multimodal and unimodal settings.

    Returns:
        umu_clf_multi_loader, umu_clf_text_loader
    """

    # -------- multimodal classification --------
    umu_clf_multi_dataset = UMUClfDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="multi",
    )

    umu_clf_multi_loader = torch.utils.data.DataLoader(
        umu_clf_multi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    # -------- unimodal / text-only classification --------
    umu_clf_text_dataset = UMUClfDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="text",
    )

    umu_clf_text_loader = torch.utils.data.DataLoader(
        umu_clf_text_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    return umu_clf_multi_loader, umu_clf_text_loader

def build_umu_gen_dataloaders(
    data_path,
    processor,
    args,
    batch_size,
    shuffle: bool = False,
):
    """
    Build UMU generation dataloaders for multimodal and unimodal settings.

    Returns:
        umu_gen_multi_loader, umu_gen_text_loader
    """

    # -------- multimodal generation --------
    umu_gen_multi_dataset = UMUGenDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="multi",
    )

    umu_gen_multi_loader = torch.utils.data.DataLoader(
        umu_gen_multi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    # -------- unimodal / text-only generation --------
    umu_gen_text_dataset = UMUGenDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="text",
    )

    umu_gen_text_loader = torch.utils.data.DataLoader(
        umu_gen_text_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    return umu_gen_multi_loader, umu_gen_text_loader

def build_clear_clf_dataloader(
    data_path,
    processor,
    args,
    batch_size,
    shuffle: bool = False,
):
    """
    Build CLEAR multimodal classification dataloader.

    Returns:
        clear_clf_loader
    """

    clear_clf_dataset = ClearClfDataset(
        data_path=data_path,
        processor=processor,
        args=args,
    )

    clear_clf_loader = torch.utils.data.DataLoader(
        clear_clf_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    return clear_clf_loader

def build_clear_gen_dataloaders(
    data_path,
    processor,
    args,
    batch_size,
    shuffle: bool = False,
):
    """
    Build CLEAR generation dataloaders.

    Returns:
        clear_gen_multi_loader, clear_gen_text_loader
    """

    # -------- multimodal generation --------
    clear_gen_multi_dataset = ClearGenDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="multi",
    )

    clear_gen_multi_loader = torch.utils.data.DataLoader(
        clear_gen_multi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    # -------- text-only generation --------
    clear_gen_text_dataset = ClearGenDataset(
        data_path=data_path,
        processor=processor,
        args=args,
        modality="text",
    )

    clear_gen_text_loader = torch.utils.data.DataLoader(
        clear_gen_text_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collator(x, processor, args),
    )

    return clear_gen_multi_loader, clear_gen_text_loader

def load_forget_dataloaders(processor, args):
    """
    Load forget-set dataloaders for different datasets.

    Returns:
        forget_clf_loaders: dict
        forget_gen_loaders: dict
    """
    base_path = args.base_path.rstrip("/") + "/"

    forget_clf_loaders = {}
    forget_gen_loaders = {}

    # =========================
    # CLEAR
    # =========================
    if args.dataset == "clear":
        # ---- classification (only multimodal) ----
        forget_clf_path = base_path + f"CLEAR/forget{args.forget_ratio:02d}_perturbed"

        forget_clf_loader = build_clear_clf_dataloader(
            data_path=forget_clf_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        forget_clf_loaders["multi"] = forget_clf_loader

        # ---- generation (multi + text) ----
        forget_gen_path = base_path + f"CLEAR/forget{args.forget_ratio:02d}_plus_tofu"

        gen_multi_loader, gen_text_loader = build_clear_gen_dataloaders(
            data_path=forget_gen_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        forget_gen_loaders["multi"] = gen_multi_loader
        forget_gen_loaders["text"] = gen_text_loader

    # =========================
    # UMU
    # =========================
    elif args.dataset == "umu":
        forget_path = base_path + f"UMU/forget_{args.forget_ratio}"

        # ---- classification (multi + text) ----
        clf_multi_loader, clf_text_loader = build_umu_clf_dataloaders(
            data_path=forget_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        forget_clf_loaders["multi"] = clf_multi_loader
        forget_clf_loaders["text"] = clf_text_loader

        # ---- generation (multi + text) ----
        gen_multi_loader, gen_text_loader = build_umu_gen_dataloaders(
            data_path=forget_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        forget_gen_loaders["multi"] = gen_multi_loader
        forget_gen_loaders["text"] = gen_text_loader

    return forget_clf_loaders, forget_gen_loaders

def load_retain_dataloaders(processor, args):
    """
    Load retain-set dataloaders for different datasets.

    Returns:
        retain_clf_loaders: dict
        retain_gen_loaders: dict
    """
    base_path = args.base_path.rstrip("/") + "/"

    retain_clf_loaders = {}
    retain_gen_loaders = {}

    retain_ratio = 100 - args.forget_ratio

    # =========================
    # CLEAR
    # =========================
    if args.dataset == "clear":
        # ---- classification (only multimodal) ----
        retain_clf_path = base_path + "CLEAR/retain_perturbed"

        retain_clf_loader = build_clear_clf_dataloader(
            data_path=retain_clf_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        retain_clf_loaders["multi"] = retain_clf_loader

        # ---- generation (multi + text) ----
        retain_gen_path = base_path + f"CLEAR/retain{retain_ratio:02d}_plus_tofu"

        gen_multi_loader, gen_text_loader = build_clear_gen_dataloaders(
            data_path=retain_gen_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        retain_gen_loaders["multi"] = gen_multi_loader
        retain_gen_loaders["text"] = gen_text_loader

    # =========================
    # UMU
    # =========================
    elif args.dataset == "umu":
        retain_path = base_path + f"UMU/retain_{retain_ratio}"

        # ---- classification (multi + text) ----
        clf_multi_loader, clf_text_loader = build_umu_clf_dataloaders(
            data_path=retain_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        retain_clf_loaders["multi"] = clf_multi_loader
        retain_clf_loaders["text"] = clf_text_loader

        # ---- generation (multi + text) ----
        gen_multi_loader, gen_text_loader = build_umu_gen_dataloaders(
            data_path=retain_path,
            processor=processor,
            args=args,
            batch_size=args.batch_size,
            shuffle=False,
        )

        retain_gen_loaders["multi"] = gen_multi_loader
        retain_gen_loaders["text"] = gen_text_loader

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return retain_clf_loaders, retain_gen_loaders
