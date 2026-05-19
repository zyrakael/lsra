import datetime
import json
import os
from typing import Literal

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from eval.score_by_llm import score_batch
from finetune.dataset import collate_fn_multimodal, collate_fn_unimodal
from utils.model_utils import MODELS
from write_log import write_logger

def data_process_clf_batch(dataloader, processor, model, args):

    model.eval()

    pred_list = []
    
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on clf")):
        question_list = sample['question']
        options_list = sample['options']
        gt_list = sample['ground_truth']
        inputs = sample['inputs']

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=24,
            min_new_tokens=1,
        )
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i in range(len(question_list)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "LLaVA" in args.model:
                pred_idx = output_text[i].find("ASSISTANT:")
                pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            # correct = check_answer(question_list[i], options_list[i], pred, sample['gt'][i])
            
            pred_list.append({
                "question": question_list[i],
                "options": options_list[i],
                "gt": gt_list[i],
                "pred": pred,
            })

        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataloader)} finished\n {json.dumps(pred_list[-1], indent=2, ensure_ascii=False)}")

    return pred_list

def data_process_gen_batch(dataloader, processor, model, args):
    model.eval()

    pred_list = []
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on gen")):
        questions = sample['question']
        ground_truths = sample['ground_truth']
        inputs = sample.get('inputs')

        assert len(questions) == len(ground_truths)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300
        )
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i in range(len(output_text)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "LLaVA" in args.model:
                pred_idx = output_text[i].find("ASSISTANT:")
                pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            pred_list.append({
                "question": questions[i],
                "gt": ground_truths[i],
                "pred": pred,
            })
        
        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataloader)} finished\n{json.dumps(pred_list[-1], indent=2)}")

    return pred_list


def filter_dataset_by_gen_eval(dataloader, dataset, processor, model, args, mode: Literal["multimodal", "unimodal"] = None):
    """
    Use the same pipeline as ``evaluate_gen`` (data_process_gen_batch +
    score_batch) to judge correctness, drop the wrongly-answered samples in
    place from ``dataset.samples`` (keep only the correctly-answered ones),
    and rebuild a new dataloader using the collate functions in
    ``finetune.dataset``.

    Args:
        dataloader: Must be built from the given dataset; each batch must
            contain ``question`` / ``ground_truth`` / ``inputs`` (or input_ids).
        dataset: Must expose a ``.samples`` list that will be filtered in place.
            If it has a ``.mode`` attribute the collate is chosen from there,
            otherwise the ``mode`` argument is used.
        mode: 'multimodal' or 'unimodal'; used only when the dataset has no
            ``.mode`` attribute.

    Returns:
        (dataloader, preds). Each prediction contains the ``correct`` flag and
        an ``id`` field if present in the input.
    """
    preds = data_process_gen_batch(dataloader, processor, model, args)
    preds, acc = score_batch(preds, "gen", args)
    keep_indices = [i for i, p in enumerate(preds) if p.get("correct")]
    dataset.samples = [dataset.samples[i] for i in keep_indices]

    model_type = MODELS.get(args.model, {}).get("type", "llava")
    if mode == "multimodal":
        collate_fn = lambda x: collate_fn_multimodal(x, processor, model_type)
    else:
        collate_fn = lambda x: collate_fn_unimodal(x, processor, model_type)
    filtered_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return filtered_loader,preds


def evaluate_clf(dataloader,
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'],
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "umu" in args.dataset:
            preds = data_process_clf_batch(dataloader, processor, model, args)
        elif "clear" in args.dataset:
            preds = data_process_clf_batch(dataloader, processor, model, args)
    elif data_modality == 'text':
        if "umu" in args.dataset:
            preds = data_process_clf_batch(dataloader, processor, model, args)
        elif "clear" in args.dataset:
            assert False
    else:
        assert False

    # Move the main model to CPU to free up GPU memory for the judge LLM,
    # avoiding OOM when two large models would otherwise coexist on the same GPU.
    model.cpu()
    torch.cuda.empty_cache()
    preds, acc_by_llm = score_batch(preds, 'clf', args)
    model.to(args.model_device)

    msg = f"{dataset_split}set Finished.\tAccuracy by LLM: {acc_by_llm:.2%}"
    print(msg)
    log_dir = args.output_file_path
    now = datetime.datetime.now().strftime('%d%H%M%S')

    log_dir = os.path.join(log_dir, args.this_run_id, now)
    os.makedirs(log_dir, exist_ok=True)

    pred_save_path = os.path.join(log_dir,f"clf_{dataset_split}set_{data_modality}_preds.json")

    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            # "acc": accuracy,
            "acc_by_llm": acc_by_llm,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 
    return acc_by_llm, log_dir

def evaluate_gen(dataloader,
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'],
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "umu" in args.dataset:
            preds = data_process_gen_batch(dataloader, processor, model, args)
        if "clear" in args.dataset:
            preds = data_process_gen_batch(dataloader, processor, model, args)
    elif data_modality == 'text':
        if "umu" in args.dataset:
            preds = data_process_gen_batch(dataloader, processor, model, args)
        if "clear" in args.dataset:
            preds = data_process_gen_batch(dataloader, processor, model, args)
    else:
        assert False
    
    # calculate rouge and bleu
    from metrics.bleu.bleu import Bleu
    from metrics.rouge.rouge import Rouge
    bleu = Bleu()
    rouge = Rouge()
    try:
        bleu_scores = bleu.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    except ZeroDivisionError:
        bleu_scores = {'bleu': 0}
    rouge_scores = rouge.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    bleumean = bleu_scores['bleu']
    rouge1mean = rouge_scores['rouge1']
    rouge2mean = rouge_scores['rouge2']
    rougeLmean = rouge_scores['rougeL']
    rougeLsummean = rouge_scores['rougeLsum']

    model.cpu()
    torch.cuda.empty_cache()

    preds, acc = score_batch(preds, 'gen', args)
    model.to(args.model_device)

    msg = f"{dataset_split}set Finished. acc: {acc:.2%}, Rouge1: {rouge1mean:.2%}, Rouge2: {rouge2mean:.2%}, RougeL: {rougeLmean:.2%}, RougeLsum: {rougeLsummean:.2%}, Bleu: {bleumean:.2%}"
    print(msg)
    # save the results
    log_dir = args.output_file_path
    now = datetime.datetime.now().strftime('%d%H%M%S')

    log_dir = os.path.join(log_dir, args.this_run_id, now)
    os.makedirs(log_dir, exist_ok=True)

    pred_save_path = os.path.join(log_dir,f"gen_{dataset_split}set_{data_modality}_preds.json")
    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            "acc": acc,
            "bleu": bleumean,
            "rouge1": rouge1mean,
            "rouge2": rouge2mean,
            "rougeL": rougeLmean,
            "rougeLsum": rougeLsummean,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 
    return rougeLmean, log_dir