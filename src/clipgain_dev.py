'''
Program adapted from CLIPScore
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
import argparse
import clip
import csv
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import generation_eval_utils
import pprint
import warnings
from packaging import version
import cv2
import base64


'''
Reads the entire video and returns the
'''
def read_video(video_path: str, total_frames: int):
    """
    Reads a video file and extracts a specified number of frames.

    Args:
        video_path (str): Path to the video file.
        total_frames (int): Number of frames to extract from the video.

    Returns:
        List[np.ndarray]: A list of selected frames (base64 encoded).
        float: Duration of the video in seconds.

    """
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file.")
    try:
        # Initialize a list to store base64 encoded frames
        base_frames = []
        
        # Read frames in a loop
        while True:
            success, frame = video.read()
            if not success:
                break  # No more frames or error occurred

            base_frames.append(frame)
        
        print(f"Number of frames input to the model is: {total_frames}")
        if total_frames == 1:
            selected_indices = [np.random.choice(range(total_frames))]
        else:
            selected_indices = np.linspace(0, len(base_frames) - 1, total_frames, dtype=int)

        selected_base_frames = [base_frames[index] for index in selected_indices]
        # print(len(selected_base_frames), "############### LEN")

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0


        return selected_base_frames, duration
    finally:
        # Release the video capture object
        video.release()


def extract_frames_from_video(video_path, frame_rate=1):
    """
    Extract frames from a video file at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Number of frames to extract per second.

    Returns:
        List[str]: List of file paths to the extracted frames.
    """
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    frame_count = 0
    while success:
        if frame_count % int(video.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            frame_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        success, frame = video.read()
        frame_count += 1
    video.release()
    return frames


class CLIPCapDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset class for preparing captions for CLIP.

    Args:
        data (List[str]): List of captions to process.
        prefix (str): Optional prefix to prepend to each caption (default is "A photo depicts").

    Returns:
        dict: A dictionary containing tokenized captions for input into CLIP.
    """
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

class FrameDataset(torch.utils.data.Dataset):
        """
        A PyTorch dataset class for processing image frames for CLIP.

        Args:
            frames (List[np.ndarray] or np.ndarray): List of image frames or a single frame.

        Returns:
            dict: A dictionary containing preprocessed image tensors for input into CLIP.
        """

        def __init__(self, frames):
            self.frames = frames if isinstance(frames, list) else [frames]
            self.preprocess = self._transform_test(224)

        def _transform_test(self, n_px):
            return Compose([
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        def __getitem__(self, idx):
            frame = self.frames[idx]
            frame = Image.fromarray(frame)
            frame = self.preprocess(frame)
            return {'image': frame}

        def __len__(self):
            return len(self.frames)

class CLIPImageDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset class for loading and preprocessing images from file paths for CLIP.

    Args:
        data (List[str]): List of image file paths.
    
    Returns:
        dict: A dictionary containing preprocessed image tensors for input into CLIP.
    """

    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=64, num_workers=8):
    """
    Encodes all captions into feature embeddings using the CLIP model.

    Args:
        captions (List[str]): List of captions to encode.
        model (torch.nn.Module): Pretrained CLIP model.
        device (str): Device to run the computations on ("cuda" or "cpu").
        batch_size (int): Batch size for processing captions.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        np.ndarray: A matrix of caption embeddings.
    """

    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(frames, model, device, batch_size=64, num_workers=8):
    """
    Encodes all image frames into feature embeddings using the CLIP model.

    Args:
        frames (List[np.ndarray]): List of image frames to encode.
        model (torch.nn.Module): Pretrained CLIP model.
        device (str): Device to run the computations on ("cuda" or "cpu").
        batch_size (int): Batch size for processing images.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        np.ndarray: A matrix of image feature embeddings.
    """
    
    data = torch.utils.data.DataLoader(
        FrameDataset(frames),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, frame_features, captions, device, w=1.0):
    """
    Calculates the CLIPScore between image features and text features.

    Args:
        model (torch.nn.Module): Pretrained CLIP model.
        frame_features (np.ndarray): Image feature embeddings.
        captions (List[str]): List of captions corresponding to the images.
        device (str): Device to run the computations on ("cuda" or "cpu").
        w (float): Weighting factor for scaling similarity scores.

    Returns:
        float: Mean CLIPScore across all image-caption pairs.
        np.ndarray: Per-image-caption similarity scores.
        np.ndarray: Caption feature embeddings.
    """


    caption_features =  extract_all_captions(captions, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        frame_features = sklearn.preprocessing.normalize(frame_features, axis=1)
        
        caption_features = sklearn.preprocessing.normalize(caption_features, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        frame_features = frame_features / np.sqrt(np.sum(frame_features**2, keepdims=True))
        caption_features = caption_features / np.sqrt(np.sum(caption_features**2, keepdims=True))

    per = w*np.clip(np.sum(frame_features * caption_features), 0, None)
    return np.mean(per), per, caption_features


# def get_refonlyclipscore(model, references, candidates, device):
#     '''
#     The text only side for refclipscore
#     '''
#     if isinstance(candidates, list):
#         candidates = extract_all_captions(candidates, model, device)

#     flattened_refs = []
#     flattened_refs_idxs = []
#     for idx, refs in enumerate(references):
#         flattened_refs.extend(refs)
#         flattened_refs_idxs.extend([idx for _ in refs])

#     flattened_refs = extract_all_captions(flattened_refs, model, device)

#     if version.parse(np.__version__) < version.parse('1.21'):
#         candidates = sklearn.preprocessing.normalize(candidates, axis=1)
#         flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
#     else:
#         warnings.warn(
#             'due to a numerical instability, new numpy normalization is slightly different than paper results. '
#             'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

#         candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
#         flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

#     cand_idx2refs = collections.defaultdict(list)
#     for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
#         cand_idx2refs[cand_idx].append(ref_feats)

#     assert len(cand_idx2refs) == len(candidates)

#     cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

#     per = []
#     for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
#         cur_refs = cand_idx2refs[c_idx]
#         all_sims = cand.dot(cur_refs.transpose())
#         per.append(np.max(all_sims))

#     return np.mean(per), per

def avg_pool_features(frame_features):

    return np.mean(frame_features, axis=0)

def get_middle_frame(selected_frames):
    if not selected_frames:
        raise ValueError("The list of frames is empty. Cannot extract the middle frame")
    
    num_frames = len(selected_frames)

    mid_idx = num_frames //2

    return selected_frames[mid_idx]

def compute_matrix_gain(single_frame_score, m_frame_score):
    score = (m_frame_score / single_frame_score) - 1
    return score


def main():
    video_dir = '/fs/classhomes/fall2024/cmsc848k/c848k004/dataset'
    video_caption_path = '/fs/classhomes/fall2024/cmsc848k/c848k004/test/onevision_video_captions.json'
    frame_caption_path = '/fs/classhomes/fall2024/cmsc848k/c848k004/test/onevision_img_captions.json'

    csv_data = []
    with open(frame_caption_path) as f:
            fdata = json.load(f)

    with open(video_caption_path) as f:
            vdata = json.load(f)

    frame_captions = list(fdata.values())
    video_captions = list(vdata.values())

    video_files = [f for f in os.listdir(video_dir)
                if f.endswith(('.mp4', '.avi','.mov'))]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        print(f"Processing {video_file}...")

        extracted_frames, durations = read_video(video_path, total_frames=8)
        single_frame = get_middle_frame(extracted_frames)    

        # print(caption)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, transform = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()

        mframe_features = extract_all_images(extracted_frames, model, device)
        video_feature = avg_pool_features(mframe_features)

        video_clip_score, _, _ = get_clip_score(
            model, video_feature, video_captions, device)
        print("Overall Video CLIPScore:", video_clip_score)


        single_frame_features = extract_all_images(single_frame, model, device)

        frame_clip_score, _, _ = get_clip_score(
            model, single_frame_features, frame_captions, device)
        print("Single Frame CLIPScore:", frame_clip_score)

        final_score = compute_matrix_gain(frame_clip_score, video_clip_score)
        print("Matrix Gain is:", final_score)

                # Append data for this video to csv_data
        csv_data.append([
            video_file,
            video_clip_score,
            frame_clip_score,
            final_score
        ])

    # Write data to CSV file
    csv_file_path = 'onevision_results.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(['Video File', 'Video CLIPScore', 'Frame CLIPScore', 'Matrix Gain'])
        
        # Write data rows
        csv_writer.writerows(csv_data)

    print(f"Results have been written to {csv_file_path}")

    


if __name__ == '__main__':
    main()
