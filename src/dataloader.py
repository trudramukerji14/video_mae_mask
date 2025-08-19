# src/data_loader.py
import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import UCF101
from torchvision.transforms import ToTensor
from ..config.load_config import load_config

def unzip_dataset(zip_file_path, extract_to_path):
    """Unzips a file and handles directory creation."""
    print(f"Unzipping {zip_file_path} to {extract_to_path}...")
    os.makedirs(extract_to_path, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("Unzipping complete.")


class VideoMAEClipProcessor(torch.nn.Module):
    """
    Applies VideoMAE-specific strided sampling and transforms to a video tensor.
    """
    def __init__(self, num_frames, stride, transform=None):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.transform = transform

    def forward(self, video_tensor):
        total_frames_in_segment = video_tensor.shape[0]
        clip_span = self.stride * (self.num_frames - 1) + 1

        if total_frames_in_segment < clip_span:
            raise ValueError(f"UCF101 segment ({total_frames_in_segment} frames) is too short "
                             f"for VideoMAE sampling ({self.num_frames} frames, stride {self.stride}). "
                             f"Needs at least {clip_span} frames.")

        possible_start_frames = list(range(0, total_frames_in_segment - clip_span + 1, 1))
        
        if not possible_start_frames:
            raise ValueError("No valid starting frames for VideoMAE clip in this segment.")

        # Randomly choose one starting point
        clip_start_frame = possible_start_frames[torch.randint(len(possible_start_frames), (1,)).item()]

        # Slice the frames using the stride
        sampled_clip = video_tensor[clip_start_frame : clip_start_frame + self.stride * self.num_frames : self.stride]

        if self.transform:
            sampled_clip = self.transform(sampled_clip)

        return sampled_clip


# The UCF101 class requires a specific file structure, but it's a robust starting point.

class CustomUCF101ForVideoMAE(UCF101):
    """
    Wraps the torchvision UCF101 class to integrate with the VideoMAEClipProcessor.
    """
    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips,
                 fold, train, video_mae_processor):
        # We pass transform=None to the parent class because our processor handles it.
        super().__init__(root, annotation_path, frames_per_clip, step_between_clips,
                         fold=fold, train=train, transform=None, output_format='TCHW')
        self.video_mae_processor = video_mae_processor

    def __getitem__(self, idx):
        # Retrieve a clip of `frames_per_clip` length from UCF101
        # The base UCF101 class handles finding the video and extracting the clip
        video, audio, label = super().__getitem__(idx)
        
        # Apply the VideoMAE specific strided sampling and transforms
        processed_video = self.video_mae_processor(video)
        
        # Return the processed video and label, as VideoMAE is a vision-only model
        return processed_video, label 


if __name__ == '__main__':
    # --- Example usage of the DataLoader module ---
    # This block is for testing the script directly.

   
   # These are just examples, please load your examples below
    zip_path = '../data/ucf101_video.zip'
    extract_path = '../data/extracted_files/'
    annotation_path = '../data/extracted_files/UCF-One Video/Annotation'
    
   
    # unzip_dataset(zip_path, extract_path) # Uncomment to run this part

    
    mae_processor = VideoMAEClipProcessor(num_frames=16, stride=4)


    try:
        dataset = CustomUCF101ForVideoMAE(
            root=extract_path,
            annotation_path=annotation_path,
            frames_per_clip=64, # A large base clip size to ensure successful sampling
            step_between_clips=1,
            fold=1,
            train=True,
            video_mae_processor=mae_processor
        )

       
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        print(f"Dataset created with {len(dataset)} clips available.")
        
        
        for video, label in dataloader:
            print(f"Sample video batch shape: {video.shape}")
            print(f"Sample label batch shape: {label.shape}")
            break # Get one batch and break
    except Exception as e:
        print(f"Could not initialize the dataset. Please ensure the path to the video file and annotation file is correct. Error: {e}")