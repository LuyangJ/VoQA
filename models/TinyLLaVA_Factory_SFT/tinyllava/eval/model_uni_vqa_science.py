import argparse
import torch
import os
import random
import re
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model_uni(model_path)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # The original file for sqa is json, but for data preparetion we have save it in jsonl
    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
         questions = [json.loads(line.strip()) for line in f if line.strip()]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    model.to(device='cuda')
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        # directions = ['l', 'r', 'u', 'd']
        # direction = random.choice(directions)
        # image_file = os.path.join(args.image_folder, idx, f'{direction}.jpg')
        if args.direction is not None:
            if args.direction == 'random':
                directions = ['l', 'r', 'u', 'd']
                direction = random.choice(directions)
            else:
                direction = args.direction
            image_file = os.path.join(args.image_folder, idx, f'{direction}.jpg')
        else:
            image_file = os.path.join(args.image_folder, f'{idx}.jpg')

        image = Image.open(image_file).convert('RGB')
        image_sizes = [image.size]
        image = image_processor(image)
        images = image.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
        question = '\n'
        # question = line['conversations'][0]
        # question = question['value'].replace('<image>', '').strip()
        # if 'image' in line:
        #     image_file = line["image"]
        #     image = Image.open(os.path.join(args.image_folder, image_file))
        #     image_sizes = [image.size]
        #     image = image_processor(image)
        #     images = image.unsqueeze(0).half().cuda()
        #     question = '<image>' + '\n' + question
        # else:
        #     images = None
        #     image_sizes = None

        if args.single_pred_prompt:
            question = question + '\n' + "Answer with the option's letter from the given choices directly."
        msg = Message()
        msg.add_message(question)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id

            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.for_benchmark:
                    match = re.search(r"ASSISTANT\s*:\s*(.*)", outputs)
                    if match:
                        result = match.group(1)
                        outputs = result.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_path.split('/')[-1],
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    ALLOWED_DIRECTIONS = ['l', 'r', 'u', 'd', 'random']
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    parser.add_argument("--direction", type=str, default=None, choices=ALLOWED_DIRECTIONS, help="Specify a direction ('l', 'r', 'u', 'd') or set to 'random' to choose a direction randomly for concat image data")
    parser.add_argument("--for_benchmark", action="store_true", help="Enable benchmark mode")
    parser.add_argument("--no_for_benchmark", action="store_false", dest="for_benchmark", help="Disable benchmark mode")
    parser.set_defaults(for_benchmark=True)      
    args = parser.parse_args()

    eval_model(args)


