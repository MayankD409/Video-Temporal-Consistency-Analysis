import os
import sys
import json
import logging
import numpy as np
import av
from tqdm import tqdm
import torch

from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, MAX_TOKENS

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str,
                      shuffle: bool = False):

    logging.info(f"Model: {model_name}")
    logging.info(f"Model {model_name} has a default total frame number of 8.")
    # If model is downloaded locally
    model_path = f"./pretrained/{model_name}"

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        "LanguageBind/Video-LLaVA-7B-hf", 
        device_map="auto",
        torch_dtype=torch.float16
    )
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    # processor.patch_size = model.config.vision_config.patch_size
    # processor.vision_feature_select_strategy = "default"
    # processor.num_additional_image_tokens = 1

    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        inp, all_choices, index2ans = construct_prompt(
            question=question,
            options=options,
            num_frames=total_frames
        )

        container = av.open(video_path)
        total_video_frames = container.streams.video[0].frames
        indices = np.arange(0, total_video_frames, total_video_frames / total_frames).astype(int)
        clip = read_video_pyav(container, indices)

        prompt = f"USER: {inp} ASSISTANT:"
        inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)

        generate_ids = model.generate(
            **inputs, 
            max_length=MAX_TOKENS, 
            do_sample=False,
            temperature=GENERATION_TEMPERATURE
        )

        response = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].replace(prompt, "").strip()

        with open(output_dir, "a") as f:
            f.write(json.dumps(
                {
                    "id": id_,
                    "question": question,
                    "response": response,
                    "all_choices": all_choices,
                    "index2ans": index2ans,
                    'gt': gt
                }
            ) + "\n")
