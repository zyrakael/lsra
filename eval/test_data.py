import os
from typing import Literal
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoTokenizer
import random
from torch.utils.data import Dataset
from datasets import load_dataset


class HFDatasetFromPath(Dataset):
    """
    Generic HuggingFace dataset loader from path.
    """
    def __init__(self, data_path, split="train"):
        self.data = load_dataset(
                "json",
                data_files={
                    "train": os.path.join(data_path, "data.json")
                },
                split="train",
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class UMUClfDataset(torch.utils.data.Dataset):
    """
    UMU Classification Dataset
    Supports unimodal / multimodal classification tasks.
    """

    def __init__(self,data_path,processor,args,modality: str = "multi"):
        assert modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"

        self.raw_dataset = HFDatasetFromPath(data_path)
        self.processor = processor
        self.data_path = data_path
        self.args = args
        self.modality = modality

        self.processed_dataset = self._process_dataset()

    def _load_image(self, image_path: str):
        """
        image_path: e.g. 'images/0001.jpg'
        return: PIL.Image
        """
        if image_path is None:
            return None

        full_path = os.path.join(self.data_path, image_path)

        image = Image.open(full_path).convert("RGB")
        image = image.resize(
            (self.args.image_resize, self.args.image_resize)
        )
        return image

    def build_llava_conversation(self,q,temp_str, modality):
        if modality == "text":
            return (
                "USER:\n"
                f"{q}\n"
                f"Select answer in {temp_str}\n"
                "ASSISTANT:"
            )
        elif modality == "multi":
            return (
                "USER: <image>\n"
                f"{q}\n"
                f"Select answer in {temp_str}\n"
                "ASSISTANT:"
            )
    
    def build_qwen_conversation(self,q, temp_str, modality, processor):
        text = (
            f"{q}\n"
            f"Select answer in {temp_str}"
        )

        if modality == "text":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
        elif modality == "multi":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]

        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _process_dataset(self):
        processed = []

        for item in self.raw_dataset:
            # -------- Image processing --------
            image = None
            if self.modality == "multi":
                image = self._load_image(item.get("image_path"))

            # -------- Load UMU classification data --------
            classify_block = item.get("Classify", {})
            if self.modality == "multi":
                raw_classify = classify_block.get("muitimodal", {})
            else:
                raw_classify = classify_block.get("unimodal", {})

            classify_data = dict(raw_classify)


            for field, field_data in classify_data.items():
                if not isinstance(field_data, dict):
                    continue
                q = field_data["question"]
                options = field_data["options"]              # dict: A/B/C/D
                gt_letter = field_data["answer"].strip().replace(".", "")
                gt_text = options[gt_letter]
                temp = [
                    options["A"],
                    options["B"],
                    options["C"],
                    options["D"],
                ]
                temp_str = str(temp)

                if "qwen" in self.args.model.lower():
                    chat_prompt = self.build_qwen_conversation(
                        q,
                        temp_str,
                        self.modality,
                        self.processor
                    )

                elif "llava" in self.args.model.lower():
                    chat_prompt = self.build_llava_conversation(
                        q,
                        temp_str,
                        self.modality
                    )
                if self.modality == "multi":
                    processed.append({
                        "question": q,
                        "ground_truth": gt_text,   
                        "options": options,
                        "image": image,            
                        "chat": chat_prompt,
                    })
                else:
                    processed.append({
                        "question": q,
                        "ground_truth": gt_text,
                        "options": options,
                        "chat": chat_prompt,
                    })

        return processed

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

class UMUGenDataset(torch.utils.data.Dataset):
    """
    UMU Generation Dataset
    Supports unimodal / multimodal generation tasks.
    """

    def __init__(self,data_path,processor,args,modality: Literal["multi", "text"] = "multi"):
        self.raw_dataset = HFDatasetFromPath(data_path)
        self.processor = processor
        self.args = args
        self.data_path = data_path
        self.modality = modality
        assert self.modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"

        self.processed_dataset = self._process_dataset()

    def _load_image(self, image_path: str):
        """
        image_path: e.g. 'images/0001.jpg'
        return: PIL.Image
        """
        if image_path is None:
            return None

        full_path = os.path.join(self.data_path, image_path)

        image = Image.open(full_path).convert("RGB")
        image = image.resize(
            (self.args.image_resize, self.args.image_resize)
        )
        return image
    
    
    def build_qwen_conversation(self,sample, modality, processor):
        q = sample["question"]

        if modality == "text":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                    ]
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q},
                    ]
                }
            ]

        chat_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        return chat_prompt

    def build_llava_conversation(self,sample, modality):
        q = sample["question"]

        if modality == "multi":
            return (
                "USER: <image>\n"
                f"{q}\n"
                "ASSISTANT:"
            )
        else:
            return (
                "USER:\n"
                f"{q}\n"
                "ASSISTANT:"
            )

    def _process_dataset(self):
        processed = []

        for i in range(len(self.raw_dataset)):
            item = self.raw_dataset[i]

            # -------- Image processing logic --------
            if self.modality == "text":
                image = None
            else:
                image_path = item.get("image_path")
                image = self._load_image(image_path)

            # -------- Load UMU generation data block --------
            gen_block = item.get("Generation", {})

            if self.modality == "multi":
                raw_gen = gen_block.get("muitimodal", {})
            else:
                raw_gen = gen_block.get("unimodal", {})

            gen_data = dict(raw_gen)


            # -------- Iterate over each generation field --------
            for field, field_data in gen_data.items():
                if not isinstance(field_data, dict):
                    continue
                q = field_data["question"]
                gt = field_data["answer"]

                # -------- Build conversation --------
                if "qwen" in self.args.model.lower():
                    chat_prompt = self.build_qwen_conversation(
                        sample={"question": q},
                        modality=self.modality,
                        processor=self.processor
                    )

                elif "llava" in self.args.model.lower():
                    chat_prompt = self.build_llava_conversation(
                        sample={"question": q},
                        modality=self.modality
                    )

                # -------- Save sample with fields aligned to CLEAR --------
                if self.modality == "multi":
                    processed.append({
                        "question": q,
                        "ground_truth": gt,
                        "image": image,
                        "chat": chat_prompt,
                    })
                else:
                    processed.append({
                        "question": q,
                        "ground_truth": gt,
                        "chat": chat_prompt,
                    })

        return processed

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

class ClearClfDataset(torch.utils.data.Dataset):
    image_caption_questions = [
    "Which option best describes the image?",
    "Which caption matches the image most accurately?",
    "Select the best description of the image from the options below.",
    "Which option correctly describes the content of the image?",
    "Choose the caption that best matches the image.",
    ]

    def __init__(self, data_path, processor, args, data_type="train"):
        self.raw_dataset = HFDatasetFromPath(data_path)
        self.processor = processor
        self.args = args
        self.data_path = data_path  
        self.data_type = data_type

        self.processed_dataset = self._process_dataset()
    
    def build_qwen_conversation(self,q: str, option_str: str, processor) -> str:
        text = (
            f"{q}\n"
            f"{option_str}\n"
            "You MUST choose one option from A to E."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]

        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

    def build_llava_conversation(self,q: str, option_str: str) -> str:
        text = (
            f"{q}\n"
            f"{option_str}\n"
            "You MUST choose one option from A to E."
        )

        return (
            "USER: <image>\n"
            f"{text}\n"
            "ASSISTANT:"
        )


    def _process_dataset(self):
        processed = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]
            image = self._load_image(sample.get("image_path"))
            q = random.choice(self.image_caption_questions) +"Answer with ONE LETTER direclty."
            captions = sample['perturbed_captions']
            gt = sample["caption"]
            o = {}
            o["A"], o["B"], o["C"], o["D"], o["E"], o["F"] = captions[0], captions[1], captions[2], captions[3], captions[4], gt
            options = {
                "A": captions[0],
                "B": captions[1],
                "C": captions[2],
                "D": captions[3],
                "E": captions[4],
                "F": gt,
            }
            option_str = "\n".join([f"{k}. {v.strip()}" for k, v in options.items()])
            # option_str = " ".join([f"{k}. {v}" for k, v in options.items()])
            if "qwen" in self.args.model.lower():
                chat_prompt = self.build_qwen_conversation(
                    q,
                    option_str,
                    processor=self.processor
                )

            elif "llava" in self.args.model.lower():
                chat_prompt = self.build_llava_conversation(
                    q,
                    option_str
                )


            processed.append({
                'question': q,
                'ground_truth': gt,
                'options': o,
                'image': image,
                'chat': chat_prompt,
            })
        return processed

    def _load_image(self, image_path: str):
        """
        image_path: e.g. 'images/0001.jpg'
        return: PIL.Image
        """
        if image_path is None:
            return None

        full_path = os.path.join(self.data_path, image_path)

        image = Image.open(full_path).convert("RGB")
        image = image.resize(
            (self.args.image_resize, self.args.image_resize)
        )
        return image

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

class ClearGenDataset(torch.utils.data.Dataset):
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

    def __init__(self, data_path, processor, args, data_type="train", modality: Literal["multi", "text"] = "multi"):
        self.raw_dataset = HFDatasetFromPath(data_path)
        self.processor = processor
        self.args = args
        self.data_path = data_path
        self.data_type = data_type

        self.modality = modality
        assert self.modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"


        self.processed_dataset = self._process_dataset()

    def build_qwen_conversation(self,sample, modality, processor):
        q = sample["question"]

        if modality == "text":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                    ]
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q},
                    ]
                }
            ]

        chat_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        return chat_prompt

    def build_llava_conversation(self,sample, modality):
        q = sample["question"]

        if modality == "multi":
            return (
                "USER: <image>\n"
                f"{q}\n"
                "ASSISTANT:"
            )
        else:
            return (
                "USER:\n"
                f"{q}\n"
                "ASSISTANT:"
            )

    def _load_image(self, image_path: str):
        """
        image_path: e.g. 'images/0001.jpg'
        return: PIL.Image
        """
        if image_path is None:
            return None
        full_path = os.path.join(self.data_path, image_path)

        image = Image.open(full_path).convert("RGB")
        image = image.resize(
            (self.args.image_resize, self.args.image_resize)
        )
        return image

    def _process_dataset(self):
        processed = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]
            image = sample.get('image')
            if self.modality == "text":
                image = None
            else:
                image = self._load_image(sample.get("image_path"))

            if self.modality == "multi":
                if sample.get("type") != "image":
                    continue
                gt = sample["caption"]
                q = random.choice(self.image_caption_questions)
            else:
                if sample.get("type") != "text":
                    continue
                gt = sample["answer"]
                q = sample["question"]

            if "qwen" in self.args.model.lower():
                chat_prompt = self.build_qwen_conversation(
                    sample={"question": q},
                    modality=self.modality,
                    processor=self.processor
                )

            elif "llava" in self.args.model.lower():
                chat_prompt = self.build_llava_conversation(
                    sample={"question": q},
                    modality=self.modality
                )
            if self.modality == "multi":
                processed.append({
                    'question': q,
                    'ground_truth': gt,
                    'image': image,
                    'chat': chat_prompt,
                })
            else:
                processed.append({
                    'question': q,
                    'ground_truth': gt,
                    'chat': chat_prompt,
                })
        return processed


    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

def collator(batch, processor, args):
    batch_data = {}
    for sample in batch:
        for key, value in sample.items():
            if key not in batch_data:
                batch_data[key] = []
            batch_data[key].append(value)
    
    batch_data['inputs'] = processor(
        text=batch_data['chat'], 
        images=batch_data.get('images') or batch_data.get('image'),
        padding=True, 
        return_tensors="pt"
    ).to(args.model_device)

    return batch_data
