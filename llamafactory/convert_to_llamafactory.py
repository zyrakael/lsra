"""Convert UMU/CLEAR datasets to LLaMAFactory ShareGPT format."""

import json
import argparse
import os
import random

# Question list used for the CLEAR dataset.
IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image.",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]


def convert_umu(data_json_path: str, output_json_path: str, include_unimodal: bool = False):
    """Convert the UMU dataset."""
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data:
        image_filename = os.path.basename(item['image_path'])

        # Multimodal QA
        mm_qa = item.get('MM_QA', {})
        questions = mm_qa.get('question', {})
        answers = mm_qa.get('answer', {})

        for key in questions:
            question, answer = questions[key], answers.get(key, "")
            if question and answer:
                result.append({
                    "messages": [
                        {"content": f"<image>{question}", "role": "user"},
                        {"content": answer, "role": "assistant"}
                    ],
                    "images": [image_filename]
                })

            # Unimodal QA
        if include_unimodal:
            um_qa = item.get('UM_QA', {})
            questions = um_qa.get('question', {})
            answers = um_qa.get('answer', {})

            for key in questions:
                question, answer = questions[key], answers.get(key, "")
                if question and answer:
                    result.append({
                        "messages": [
                            {"content": question, "role": "user"},
                            {"content": answer, "role": "assistant"}
                        ],
                        "images": []
                    })

    _save_result(result, output_json_path)
    return result


def convert_clear(data_json_path: str, output_json_path: str):
    """Convert the CLEAR dataset, supporting both image and text item types."""
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data:
        item_type = item.get('type')
        
        if item_type == 'image':
            # Image item: use the caption as the answer.
            image_path = item.get('image_path')
            caption = item.get('caption', '')
            if not image_path or not caption:
                continue
            image_filename = os.path.basename(image_path)
            question = item.get('question') or random.choice(IMAGE_CAPTION_QUESTIONS)
            result.append({
                "messages": [
                    {"content": f"<image>{question}", "role": "user"},
                    {"content": caption, "role": "assistant"}
                ],
                "images": [image_filename]
            })
            
        elif item_type == 'text':
            # Text-only item: QA pair.
            question = item.get('question', '')
            answer = item.get('answer', '')
            if question and answer:
                result.append({
                    "messages": [
                        {"content": question, "role": "user"},
                        {"content": answer, "role": "assistant"}
                    ],
                    "images": []
                })

    _save_result(result, output_json_path)
    return result


def _save_result(result, output_json_path):
    os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"转换完成: {len(result)} 条样本 -> {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="数据集转换为 LLaMAFactory 格式")
    parser.add_argument("--dataset", type=str, required=True, choices=["umu", "clear"])
    parser.add_argument("--data_json", type=str, required=True, help="data.json 路径")
    parser.add_argument("--output_json", type=str, required=True, help="输出路径")
    parser.add_argument("--include_unimodal", action="store_true", help="包含单模态数据")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.dataset == "umu":
        convert_umu(args.data_json, args.output_json, args.include_unimodal)
    else:
        convert_clear(args.data_json, args.output_json)


if __name__ == "__main__":
    main()
