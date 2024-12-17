# evaluate.py
import os
import json
import argparse
import warnings
import importlib
from collections import defaultdict

def validate_choices(input_value, all_choices, input_name):
    if input_value == 'ALL':
        return all_choices
    else:
        selected_values = [item.strip() for item in input_value.split(",")]
        invalid_values = [item for item in selected_values if item not in all_choices]
        if invalid_values:
            raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
                             f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
        return selected_values

def load_queries(reasoning_t, demonstration_t):
    dataset_path = f"./data/{reasoning_t}.json"
    with open(dataset_path, "r") as f:
        qas = json.load(f)
    return qas, demonstration_t, reasoning_t

if __name__ == "__main__":
    # load configs
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    model_families = config.get('models', {})
    video_models = config.get('video_models', [])
    model_choices = [item for sublist in model_families.values() for item in sublist]
    reasoning_type_choices = config.get('reasoning_types', [])
    demonstration_type_choices = config.get('demonstration_types', [])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=model_choices, required=True)
    parser.add_argument('--reasoning_type', type=str, default='ALL')
    parser.add_argument('--demonstration_type', type=str, default='ALL')
    args = parser.parse_args()

    reasoning_type = validate_choices(args.reasoning_type, reasoning_type_choices, 'reasoning')
    demonstration_type = validate_choices(args.demonstration_type, demonstration_type_choices, 'demonstration')

    model_name = args.model

    # If model is a video model that processes entire video without frame sampling:
    # we only do -1 (full video)
    if model_name in video_models:
        FRAME_SUBSETS = [-1]
    else:
        # Assuming total_frames = 16 for your dataset:
        # 25% = 4 frames, 50% = 8 frames, 75% = 12 frames, 100% = 16 frames
        FRAME_SUBSETS = [4, 8, 12, 16]

    def run_inference_for_frames(model_name, model_families, qas, reasoning_t, demonstration_t, total_frames):
        output_subdir = '+'.join([reasoning_t, demonstration_t])
        output_path = f"./results/{output_subdir}/{total_frames}/{model_name}.jsonl"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        family_found = None
        for family, model_list in model_families.items():
            if model_name in model_list:
                family_found = family
                break
        if family_found is None:
            raise ValueError(f"Model {model_name} not found in any family.")

        module = importlib.import_module(f"generate_lib.{family_found}")
        generate_response = getattr(module, "generate_response")

        curr_results = set()
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                for line in f:
                    curr_results.add(json.loads(line)['id'])

        queries_to_run = defaultdict(list)
        for id_, qa in qas.items():
            if qa['demonstration_type'] == demonstration_t:
                if id_ in curr_results:
                    continue
                qa['id'] = id_
                queries_to_run[output_path].append(qa)

        if not queries_to_run[output_path]:
            print(f"No new queries to run for {reasoning_t}+{demonstration_t} at {total_frames} frames.")
            return

        print(f"Running inference for {model_name}, reasoning={reasoning_t}, demonstration={demonstration_t}, frames={total_frames}")
        generate_response(
            model_name=model_name,
            queries=queries_to_run[output_path],
            total_frames=total_frames,
            output_dir=output_path,
            shuffle=False
        )

    for rt in reasoning_type:
        for dt in demonstration_type:
            qas, dt_cur, rt_cur = load_queries(rt, dt)
            for tf in FRAME_SUBSETS:
                run_inference_for_frames(model_name, model_families, qas, rt, dt, tf)

    print("All inferences completed.")

# # evaluate.py
# import os
# import json
# import argparse
# import warnings
# import importlib
# from collections import defaultdict
# import cv2

# def validate_choices(input_value, all_choices, input_name):
#     if input_value == 'ALL':
#         return all_choices
#     else:
#         selected_values = [item.strip() for item in input_value.split(",")]
#         invalid_values = [item for item in selected_values if item not in all_choices]
#         if invalid_values:
#             raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
#                              f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
#         return selected_values

# def load_queries(reasoning_t, demonstration_t):
#     dataset_path = f"./data/{reasoning_t}.json"
#     with open(dataset_path, "r") as f:
#         qas = json.load(f)
#     return qas, demonstration_t, reasoning_t

# def get_video_frame_count(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video file: {video_path}")
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return frame_count

# def run_inference_for_frames(model_name, model_families, qas, reasoning_t, demonstration_t, frame_subsets):
#     # frame_subsets is a list of integers (frame counts) or [-1]

#     # create output path template
#     output_subdir = '+'.join([reasoning_t, demonstration_t])

#     # identify model family
#     family_found = None
#     for family, model_list in model_families.items():
#         if model_name in model_list:
#             family_found = family
#             break
#     if family_found is None:
#         raise ValueError(f"Model {model_name} not found in any family.")

#     module = importlib.import_module(f"generate_lib.{family_found}")
#     generate_response = getattr(module, "generate_response")

#     # Check existing results and skip already done queries
#     # We'll run once per frame subset
#     for total_frames in frame_subsets:
#         output_path = f"./results/{output_subdir}/{total_frames}/{model_name}.jsonl"
#         if not os.path.exists(os.path.dirname(output_path)):
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         curr_results = set()
#         if os.path.exists(output_path):
#             with open(output_path, 'r') as f:
#                 for line in f:
#                     curr_results.add(json.loads(line)['id'])

#         queries_to_run = defaultdict(list)
#         for id_, qa in qas.items():
#             if qa['demonstration_type'] == demonstration_t:
#                 if id_ in curr_results:
#                     continue
#                 qa['id'] = id_
#                 queries_to_run[output_path].append(qa)

#         if not queries_to_run[output_path]:
#             print(f"No new queries to run for {reasoning_t}+{demonstration_t} at {total_frames} frames.")
#             continue

#         print(f"Running inference for {model_name}, reasoning={reasoning_t}, demonstration={demonstration_t}, frames={total_frames}")
#         generate_response(
#             model_name=model_name, 
#             queries=queries_to_run[output_path], 
#             total_frames=total_frames, 
#             output_dir=output_path,
#             shuffle=False
#         )

# if __name__ == "__main__":
#     # load configs
#     config_path = os.path.join(os.path.dirname(__file__), 'config.json')
#     with open(config_path, 'r') as config_file:
#         config = json.load(config_file)

#     model_families = config.get('models', {})
#     video_models = config.get('video_models', [])
#     model_choices = [item for sublist in model_families.values() for item in sublist]
#     reasoning_type_choices = config.get('reasoning_types', [])
#     demonstration_type_choices = config.get('demonstration_types', [])

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, choices=model_choices, required=True)
#     parser.add_argument('--reasoning_type', type=str, default='ALL')
#     parser.add_argument('--demonstration_type', type=str, default='ALL')
#     args = parser.parse_args()

#     reasoning_type = validate_choices(args.reasoning_type, reasoning_type_choices, 'reasoning')
#     demonstration_type = validate_choices(args.demonstration_type, demonstration_type_choices, 'demonstration')

#     model_name = args.model

#     # If model is a video model, just set FRAME_SUBSETS = [-1]
#     # Otherwise we do percentage-based: [0.25,0.5,0.75,1.0]
#     PERCENT_SUBSETS = [0.25, 0.5, 0.75, 1.0]

#     for rt in reasoning_type:
#         for dt in demonstration_type:
#             qas, dt_cur, rt_cur = load_queries(rt, dt)
            
#             if model_name in video_models:
#                 # Only full video
#                 FRAME_SUBSETS = [-1]
#             else:
#                 # We must determine frame counts from percentages
#                 # We pick the first query to find a representative video
#                 # If no queries, skip
#                 if not qas:
#                     print(f"No queries for {rt}+{dt}, skipping.")
#                     continue

#                 # Get first query's video path
#                 first_id = next(iter(qas.keys()))
#                 first_qa = qas[first_id]
#                 video_path = os.path.join('videos', first_qa['demonstration_type'], first_qa['key'] + '.mp4')

#                 total_frames_video = get_video_frame_count(video_path)
#                 # Convert percentages to frame counts (rounding)
#                 FRAME_SUBSETS = [max(1, int(total_frames_video * p)) for p in PERCENT_SUBSETS]

#             run_inference_for_frames(model_name, model_families, qas, rt, dt, FRAME_SUBSETS)

#     print("All inferences completed.")
