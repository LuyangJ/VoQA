import os
import json
import argparse
from process_answer import extract_answer, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--split_word", type=str, default='ASSISTANT')
parser.add_argument("--model_type", type=str, default='zero-shot')
args = parser.parse_args()

all_answers = []
for line_idx, line in enumerate(open(args.src)):
    res = json.loads(line)
    question_id = res['question_id']
    # text = res['text'].rstrip('.').lower()
    text = res['text'].rstrip('.')
    text = extract_answer(text, args.filter_answer, args.split_word, 'gqa', args.model_type)
    text = text.lower()
    all_answers.append({"questionId": question_id, "prediction": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)
