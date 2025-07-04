import os
import json
import argparse
from process_answer import extract_answer, str2bool

def compute_edit_distance(ref: str, pred: str) -> float:
    """
    Calculate the edit distance between the two strings and regress the normalized similarity score

    Args:
        ref: Reference string
        pred: Predicted string

    Returns:
        float: 1 - Edit distance/reference string length
    """
    m, n = len(ref), len(pred)
    # Create a dynamic programming matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the first row and the first column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill the dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # delete
                    dp[i][j-1] + 1,    # insert
                    dp[i-1][j-1] + 1   # replace
                )
    
    # Calculate the normalized similarity score
    edit_distance = dp[m][n]
    if len(ref) == 0:
        return 0.0
    return 1 - edit_distance / len(ref)


def eval_pope(answers, label_file, args):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    question_list = [json.loads(q)['text'] + " Answer the question using a single word or phrase." for q in open(label_file, 'r')]

    pred_question_list = []
    for answer in answers:
        parts = answer['text'].split(args.split_word + ':')
        if len(parts) > 2:
            pred_question = parts[1].strip()
        else:
            pred_question = parts[0].strip()    

        pred_question_list.append(pred_question)
        
        text = extract_answer(answer['text'], args.filter_answer, args.split_word, 'pope', args.model_type)

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    TP_sim, TN_sim, FP_sim, FN_sim = 0, 0, 0, 0
    for i, (pred, label) in enumerate(zip(pred_list, label_list)):
        similarity_score = compute_edit_distance(question_list[i], pred_question_list[i])

        # modified
        if similarity_score < 0 :
            # It indicates that it is not a problem, and the current predicted problem length is greater than the actual problem length
            # print(similarity_score, question_list[i])
            similarity_score = 0

        if pred == pos and label == pos:
            TP += 1
            TP_sim += similarity_score
        elif pred == pos and label == neg:
            FP += 1
            FP_sim += similarity_score
        elif pred == neg and label == neg:
            TN += 1
            TN_sim += similarity_score
        elif pred == neg and label == pos:
            FN += 1
            FN_sim += similarity_score

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
    correct_sim = (TP_sim + TN_sim) / (TP + TN) if (TP + TN) > 0 else 0
    incorrect_sim = (FP_sim + FN_sim) / (FP + FN) if (FP + FN) > 0 else 0
    
    print('The average similarity of the correct answers: {:.3f}'.format(correct_sim))
    print('The average similarity of the incorrect answers: {:.3f}'.format(incorrect_sim))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    return acc, correct_sim, incorrect_sim


def cal_pope_acc(acc_dict):
    final_acc = 0
    for category, acc in acc_dict.items():
        if category == 'random':
            final_acc += 2910 * acc
        else:
            final_acc += 3000 * acc
    return final_acc / 8910.0


if __name__ == "__main__":
    print("################ POPE ###############")
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--split_word", type=str, default='ASSISTANT')
    parser.add_argument("--model_type", type=str, default='zero-shot')
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    acc_dict = {}
    correct_dict = {}
    incorrect_dict = {}
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        acc, correct_sim, incorrect_sim = eval_pope(cur_answers, os.path.join(args.annotation_dir, file), args)
        print("====================================")
        acc_dict[category] = acc
        correct_dict[category] = correct_sim
        incorrect_dict[category] = incorrect_sim
    
    # cal final acc
    final_acc = cal_pope_acc(acc_dict)
    print(f"Weighted average accuracy:", final_acc)
    final_correct = cal_pope_acc(correct_dict)
    final_incorrect = cal_pope_acc(incorrect_dict)
    print('The average similarity of the correct responses after weighting: {:.3f}'.format(final_correct))
    print('The average similarity of the incorrect responses after weighting: {:.3f}'.format(final_incorrect))
