import os
import argparse
import json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='./playground/Instructions_slim/ImageNet/test.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA')
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def is_correct_prediction(prediction, ground_truth):
    prediction = prediction.strip()
    return bool(prediction) and prediction.upper() in ground_truth.upper()


def eval_single(test_file, result_file, output_dir=None, max_samples=None):
    with open(test_file, "r") as handle:
        annotations = json.load(handle)
    if max_samples is not None:
        annotations = annotations[:max_samples]
    with open(result_file, "r") as handle:
        results = [json.loads(line) for line in handle if line.strip()]

    results_by_id = {}
    for result in results:
        question_id = str(result["question_id"])
        if question_id in results_by_id:
            raise ValueError(f"Duplicate prediction for question_id={question_id}")
        results_by_id[question_id] = result

    total = len(annotations)
    right = 0
    false_answers = []
    empty_predictions = 0
    missing_predictions = 0
    for annotation in tqdm(annotations):
        question_id = str(annotation["question_id"])
        ground_truth = annotation["answer"]
        result = results_by_id.get(question_id)
        if result is None:
            missing_predictions += 1
            result = {
                "question_id": annotation["question_id"],
                "text": "",
                "error": "missing prediction",
            }
        prediction = result.get("text", "")
        if not prediction.strip():
            empty_predictions += 1

        if is_correct_prediction(prediction, ground_truth):
            right += 1
        else:
            result["ground_truth"] = ground_truth
            false_answers.append(result)

    accuracy = 100. * right / total if total else 0.0
    summary = (
        f"Samples: {total}\n"
        f"Correct: {right}\n"
        f"Empty predictions: {empty_predictions}\n"
        f"Missing predictions: {missing_predictions}\n"
        f"Accuracy: {accuracy:.2f}%\n"
    )
    print(summary)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'Result.text')
        with open(output_file, 'w') as handle:
            handle.write(summary)
            json.dump(false_answers, handle, indent=4)


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(
            args.test_file,
            args.result_file,
            args.output_dir,
            args.max_samples,
        )
