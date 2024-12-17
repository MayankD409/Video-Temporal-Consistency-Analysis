# compute_tcs.py
import os
import json
import argparse
import re
from collections import defaultdict

def load_predictions(file_path):
    preds = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            preds[data['id']] = data
    return preds

def extract_chosen_option(response):
    # Extract chosen option from response text
    # Adjust regex as needed to match your response pattern
    match = re.search(r"answer is ([A-F])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # fallback if not found
    # try another pattern, e.g., "Therefore, the answer is X"
    match = re.search(r"Therefore, the answer is ([A-F])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def parse_gt(gt_str):
    # gt is like "C. 3"
    parts = gt_str.split('.')
    option = parts[0].strip()
    return option, None

def compute_temporal_metrics(prediction_sets):
    """
    prediction_sets: list of (frames, preds) sorted by frames
    e.g. [(8, preds_at_8), (16, preds_at_16)]
    """
    prediction_sets = sorted(prediction_sets, key=lambda x: x[0])
    all_ids = set()
    for _, p in prediction_sets:
        all_ids.update(p.keys())

    results = {}
    for qid in all_ids:
        seq = []
        for (frames, preds) in prediction_sets:
            if qid not in preds:
                # Missing prediction means no run or skipped
                seq.append((frames, None, None))
                continue
            record = preds[qid]
            chosen_option = extract_chosen_option(record['response'])
            gt_option, _ = parse_gt(record['gt'])
            seq.append((frames, chosen_option, gt_option))

        # Compute changes, time-to-correct
        changes = 0
        for i in range(1, len(seq)):
            if seq[i][1] is not None and seq[i-1][1] is not None:
                if seq[i][1] != seq[i-1][1]:
                    changes += 1

        # time-to-correct: first frames where chosen_option == gt_option
        ttc = None
        for (frames, pred_opt, g_opt) in seq:
            if pred_opt == g_opt and pred_opt is not None:
                ttc = frames
                break

        final_correct = (seq[-1][1] == seq[-1][2]) if seq[-1][1] is not None else False

        results[qid] = {
            "num_changes": changes,
            "time_to_correct": ttc if ttc is not None else None,
            "final_correct": final_correct
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name, e.g. gemini-1.5-flash")
    parser.add_argument('--reasoning_type', type=str, required=True, help="e.g. count")
    parser.add_argument('--demonstration_type', type=str, required=True, help="e.g. human")
    parser.add_argument('--frames', type=str, required=True,
                        help="Comma separated list of frame subsets e.g. 8,16 to consider.")
    args = parser.parse_args()

    frames_list = [f.strip() for f in args.frames.split(',')]
    # Load predictions for each frames subset
    prediction_sets = []
    base_dir = f"./results/{args.reasoning_type}+{args.demonstration_type}"

    for frames in frames_list:
        path = os.path.join(base_dir, frames, f"{args.model}.jsonl")
        if os.path.exists(path):
            preds = load_predictions(path)
            # Convert frames to a number for sorting
            try:
                frm_val = int(frames)
            except ValueError:
                # if -1 or any special token, handle separately
                frm_val = 9999 if frames == '-1' else 0
            prediction_sets.append((frm_val, preds))
        else:
            print(f"Warning: {path} not found, skipping.")
    
    if len(prediction_sets) < 2:
        print("Not enough prediction sets to compute temporal metrics.")
        exit(0)

    metrics = compute_temporal_metrics(prediction_sets)
    output_path = os.path.join(base_dir, f"{args.model}_temporal_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Temporal metrics saved to {output_path}.")
