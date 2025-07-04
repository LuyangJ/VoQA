# The script for analyse GQA, TextVQA and SQA. 
import argparse
import json

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


def process_sqa(sqa_pred_answers, sqa_inference_answers, save_file):
    question_id_dict = {
        'correct': [],
        'incorrect': []
    }
    
    with open(sqa_pred_answers, 'r') as f:
        data = json.load(f)
        
    for result_type in ['correct', 'incorrect']:
        for item in data[result_type]:
            question_id_dict[result_type].append(item['question_id'])
    
    similarity_scores = {
        'correct': {},
        'incorrect': {}
    }

    with open(sqa_inference_answers, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = data['question_id']
            
            real_question = data['prompt'].replace('<image>\n', '').replace('\n', ' ').strip()
            
            assistant_parts = data['text'].split('ASSISTANT:')
            if len(assistant_parts) == 2:
                pred_question = assistant_parts[0].strip()
            elif len(assistant_parts) == 3:
                pred_question = assistant_parts[1].strip()
            else:
                pred_question = data['text']
            
            similarity = compute_edit_distance(real_question, pred_question)

            if similarity < 0 :
                # It indicates that it is not a problem, and the current predicted problem length is greater than the actual problem length
                # print(similarity, data['text'])
                similarity = 0
            
            if question_id in question_id_dict['correct']:
                similarity_scores['correct'][question_id] = similarity
            elif question_id in question_id_dict['incorrect']:
                similarity_scores['incorrect'][question_id] = similarity

    correct_avg = sum(similarity_scores['correct'].values()) / len(similarity_scores['correct']) if similarity_scores['correct'] else 0
    incorrect_avg = sum(similarity_scores['incorrect'].values()) / len(similarity_scores['incorrect']) if similarity_scores['incorrect'] else 0

    print(f"The average similarity of identifying questions with correct answers: {correct_avg:.4f}")
    print(f"The average similarity of identifying questions with incorrect answers: {incorrect_avg:.4f}")
    
    if save_file:
        with open(save_file, 'w') as f:
            json.dump(similarity_scores, f, indent=2)


def process_textvqa(textvqa_question_id_scores_list, textvqa_inference_answers, save_file):
    # Read the mapping file from the score to the problem ID
    with open(textvqa_question_id_scores_list, 'r') as f:
        score_to_qids = json.load(f)
    
    # Create a mapping from the problem ID to the score
    qid_to_score = {}
    
    for score_str in score_to_qids:
        score = round(float(score_str), 1)
        qids = score_to_qids[score_str]
        # Map each question ID to its score
        for qid in qids:
            qid_to_score[qid] = score
    
    similarity_scores = {str(score): {} for score in set(qid_to_score.values())}
    
    with open(textvqa_inference_answers, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = data['question_id']
            
            if question_id in qid_to_score:
                real_question = data['prompt'].replace('<image>\n', '').replace('\n', ' ').strip()
                
                parts = data['text'].split('ASSISTANT:')
                if len(parts) > 2:  # two 'ASSISTANT'
                    pred_question = parts[1].strip()
                else:  # one 'ASSISTANT'
                    pred_question = parts[0].strip()
                
                similarity = compute_edit_distance(real_question, pred_question)

                # modified
                if similarity < 0 :
                    # It indicates that it is not a problem, and the current predicted problem length is greater than the actual problem length
                    # print(similarity, data['text'])
                    similarity = 0
                
                score = str(qid_to_score[question_id])
                similarity_scores[score][question_id] = similarity

    high_scores = {}  # >0.5
    low_scores = {}   # <=0.5
    
    for score in similarity_scores:
        score_float = float(score)
        scores = similarity_scores[score]
        if score_float > 0.5:
            high_scores.update(scores)
        else:
            low_scores.update(scores)
    
    high_avg = sum(high_scores.values()) / len(high_scores) if high_scores else 0
    low_avg = sum(low_scores.values()) / len(low_scores) if low_scores else 0
    
    print(f"Average similarity of problem identification with a score > 0.5: {high_avg:.4f}")
    print(f"Average similarity of problem identification with a score <= 0.5: {low_avg:.4f}")
    
    # Save the result to a file
    output = {
        "similarity_by_score": similarity_scores,
        "high_score_average": high_avg,
        "low_score_average": low_avg,
        "high_scores": high_scores,
        "low_scores": low_scores
    }
    
    if save_file:
        with open(save_file, 'w') as f:
            json.dump(output, f, indent=2)
    


def process_gqa(reference_file, prediction_file, inference_file):

    def evaluate_predictions(reference_path, prediction_path):
        # Read the reference answer
        with open(reference_path, 'r') as f:
            reference_data = json.load(f)
        
        # Read the prediction result
        with open(prediction_path, 'r') as f:
            predictions = json.load(f)
        
        # Convert the prediction results into a dictionary for easy lookup
        pred_dict = {item['questionId']: item['prediction'].strip().lower() for item in predictions}
        
        # Used for storing correct and incorrect problem ids
        question_id_dict = {
            'correct': [],
            'incorrect': []
        }
        total = 0

        for qid, ref in reference_data.items():
            if qid in pred_dict:
                total += 1
                gt_answer = ref['answer'].strip().lower()
                pred_answer = pred_dict[qid]
                if gt_answer == pred_answer:
                    question_id_dict['correct'].append(qid)
                else:
                    question_id_dict['incorrect'].append(qid)
        
        accuracy = len(question_id_dict['correct']) / total if total > 0 else 0.0

        print(f"Total evaluated: {total}")
        print(f"Correct: {len(question_id_dict['correct'])}")
        print(f"Incorrect: {len(question_id_dict['incorrect'])}")
        print(f"Accuracy: {accuracy:.2%}")

        return question_id_dict

    question_id_dict = evaluate_predictions(reference_file, prediction_file)

    with open(inference_file, 'r') as f:
        inference_data = [json.loads(line) for line in f]

    inference_dict = {}
    for item in inference_data:
        parts = item['text'].split('ASSISTANT:')
        if len(parts) > 2:  # multiple 'ASSISTANT'
            text = parts[1].strip()
        else:
            text = parts[0].strip()
        inference_dict[item['question_id']] = {
            'prompt': item['prompt'],
            'text': text
        }

    correct_distances = []
    incorrect_distances = []

    for qid in question_id_dict['correct']:
        if qid in inference_dict:
            distance = compute_edit_distance(
                inference_dict[qid]['prompt'],
                inference_dict[qid]['text']
            )

            # modified
            if distance < 0 :
                # It indicates that it is not a problem, and the current predicted problem length is greater than the actual problem length
                # print(distance, inference_dict[qid]['prompt'])
                distance = 0

            correct_distances.append(distance)

    for qid in question_id_dict['incorrect']:
        if qid in inference_dict:
            distance = compute_edit_distance(
                inference_dict[qid]['prompt'],
                inference_dict[qid]['text']
            )

            # modified
            if distance < 0 :
                # It indicates that it is not a problem, and the current predicted problem length is greater than the actual problem length
                # print(distance, inference_dict[qid]['prompt'])
                distance = 0

            incorrect_distances.append(distance)

    avg_correct_distance = sum(correct_distances) / len(correct_distances) if correct_distances else 0
    avg_incorrect_distance = sum(incorrect_distances) / len(incorrect_distances) if incorrect_distances else 0

    print(f"Average edit distance for correct predictions: {avg_correct_distance:.4f}")
    print(f"Average edit distance for incorrect predictions: {avg_incorrect_distance:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='task name for analysis')
    # for SQA
    parser.add_argument("--sqa_resoning_json", type=str, help='SQA original reasoning result path')
    parser.add_argument("--sqa_prediction_json", type=str, help='SQA prediction result path')
    parser.add_argument("--sqa_save_file", type=str, help='SQA analysis results save path')
    # for TextVQA
    parser.add_argument("--textvqa_id_to_score_json", type=str, help='textvqa reasoning id to score json path')
    parser.add_argument("--textvqa_prediction_json", type=str, help='textvqa prediction result path')
    parser.add_argument("--textvqa_save_file", type=str, help='textvqa analysis results save path')
    # for GQA
    parser.add_argument("--gqa_reference_json", type=str, help='GQA annotation json')
    parser.add_argument("--gqa_reasoning_json", type=str, help='GQA original reasoning result path')
    parser.add_argument("--gqa_prediction_json", type=str, help='SQA prediction result path')

    args = parser.parse_args()

    if args.task == 'scienceqa':
        print("################ SQA ###############")
        process_sqa(args.sqa_prediction_json, args.sqa_resoning_json, args.sqa_save_file)
    elif args.task == 'textvqa':
        print("################ TextVQA ###############")
        process_textvqa(args.textvqa_id_to_score_json, args.textvqa_prediction_json, args.textvqa_save_file)
    else:
        print("################ GQA ###############")
        process_gqa(args.gqa_reference_json, args.gqa_prediction_json, args.gqa_reasoning_json)
