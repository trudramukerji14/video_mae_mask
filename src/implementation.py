#src/implementation.py

import torch
import torch.nn as nn
import os
import yaml

from torch.utils.data import DataLoader
from torchvision.datasets import UFC101
from torchvision.transforms import ToTensor
from torchvision import transforms
from ..config.load_config import load_config


class VideoMAEClipProcessor(torch.nn.Module):

    def __init__(self, num_frames, stride, transform = None):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.transform = transform

    def forward(self, video_tensor):
        total_frames_in_segement = video_tensor.shape[0]
        clip_span = self.stride * (self.num_frames - 1) + 1
        if total_frames_in_segement < clip_span:
            raise ValueError(f'UCF101 segment is too short')
        possible_start_frames = list(range(0, total_frames_in_segment - clip_span + 1, 1))
        if not possible_start_frames:
            raise ValueError("No valid starting frames for VideoMAE clip in this segment.")
        clip_start_frame = possible_start_frames[torch.randint(len(possible_start_frames), (1,)).item()]
        sampled_clip = video_tensor[clip_start_frame : clip_start_frame + self.stride * self.num_frames : self.stride]
        if self.transform:
            sampled_clip = self.transform(sampled_clip)
        return sampled_clip

class CustomUCF101ForVideoMAE(UCF101):
    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips,
                 fold, train, video_mae_processor):
        super().__init__(root, annotation_path, frames_per_clip, step_between_clips,
                         fold=fold, train=train, transform=None, output_format = 'TCHW')
        self.video_mae_processor = video_mae_processor

    def __getitem__(self, idx):
        video, audio, label = super().__getitem__(idx)
        processed_video = self.video_mae_processor(video)
        return processed_video, label

class VideoMAE_mask_one_video(torch.nn.Module):
    def __init__(self, in_channels=3,
                 temporal_patch_size=2, spatial_patch_size=16,
                 embed_dim=768, num_frames=16, height=224, width=224,
                 depth=12, num_heads=12):
        super().__init__()
        
        # Define the patch sizes
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
 
        # Calculate the dimension of a single tubelet (T_patch * H_patch * W_patch * C)
        tubelet_dim = temporal_patch_size * spatial_patch_size * spatial_patch_size * in_channels
        self.tubelet_projector = nn.Linear(tubelet_dim, embed_dim)
        

        # The number of patches (tokens) is calculated here
        num_temporal_patches = num_frames // temporal_patch_size
        num_spatial_patches = (height // spatial_patch_size) * (width // spatial_patch_size)
        #num_tokens = num_temporal_patches * num_spatial_patches

    def _mask_tokens(self, tokens, mask_ratio=0.95):
        """
        Implements the masking logic from the notebook.
        
        Args:
            tokens (torch.Tensor): The embedded tokens of shape (num_tokens, embed_dim).
                                    Note: Batch dimension is handled outside for simplicity here.
            mask_ratio (float): The ratio of tokens to mask.
            
        Returns:
            visible_tokens (torch.Tensor): The tokens that are not masked.
            mask_full (torch.Tensor): A boolean mask indicating which tokens were kept.
        """
        spatial_dim, temporal_dim, embed_dim = tokens.shape[0], tokens.shape[1], tokens.shape[2]

        #create mask
        keep = 1.0 - mask_ratio

        #keep mask
        spatial_mask = torch.rand(spatial_dim) < keep #boolean tensor of shape 196
        mask_full = spatial_mask.unsqueeze(1).repeat(1, temporal_dim) #expand along time [196, 8]
        mask_flat = mask_full.T.flatten() #1568

        tokens_flat = tokens.permute(1,0,2).reshape(-1, embed_dim) #shape: [196, 8, 768] -> [8, 196, 768] -> [1568, 768]
        visible_tokens = tokens_flat[mask_flat]

        return visible_tokens, mask_full
    
        # num_temporal_patches = self.pos_embed.shape[1] // 196
        # num_spatial_patches = 196
        
        # tokens_grid = tokens.view(B, num_spatial_patches, num_temporal_patches, embed_dim)
        
        # keep = 1.0 - mask_ratio
        
        # spatial_mask = torch.rand(B, num_spatial_patches) < keep
        
        # mask_full = spatial_mask.unsqueeze(2).repeat(1, 1, num_temporal_patches)
        # mask_flat = mask_full.flatten(start_dim=1)
        
        # tokens_reshaped = tokens_grid.view(B, -1, embed_dim)
        
        # visible_tokens = tokens_reshaped[mask_flat]

        # return visible_tokens, mask_full


    def forward(self, clip, mask_ratio=0.9):
        clip = clip.squeeze(0) #remove the batch dimension: get first batch
        clip = clip.permute(1, 0, 2, 3).contiguous()  # -> [3, 16, 224, 224]
    

        # Unfold into tubelets
        tubelets = clip.unfold(1, 2, 2).unfold(2, 16, 16).unfold(3, 16, 16) #-> [3, 8, 14, 14, 2, 16, 16]
    

        # Rearrange for projection
        tubelets = tubelets.permute(1, 2, 3, 0, 4, 5, 6).contiguous()  # -> [8,14,14,3,2,16,16]
        tubelets = tubelets.view(-1, 3, 2, 16, 16)  # [1568, 3, 2, 16, 16]

        # Flatten tubelets
        tubelets_flat = tubelets.view(tubelets.size(0), -1)  # [1568, 1536]

        # Linear projection
        tokens = self.tubelet_projector(tubelets_flat)

        visible_tokens, mask_full = self._mask_tokens(tokens, mask_ratio)
        
        return visible_tokens, mask_full

        # tokens = tokens.view(196, 8, 768)  # Final shape: [196, 8, 768] since 1568 = 196 * 8
        #         # x shape: (T, C, H, W)
        # B, T, C, H, W = x.shape
        
        # x = x.permute(0, 2, 1, 3, 4).contiguous()

        # tubelets = x.unfold(2, self.temporal_patch_size, self.temporal_patch_size) \
        #             .unfold(3, self.spatial_patch_size, self.spatial_patch_size) \
        #             .unfold(4, self.spatial_patch_size, self.spatial_patch_size)
        
        # tubelets = tubelets.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        # tubelets_flat = tubelets.view(B, -1, self.tubelet_projector.in_features)

        # tokens = self.tubelet_projector(tubelets_flat)

        # tokens = tokens + self.pos_embed

        # visible_tokens, mask_full = self._mask_tokens(tokens, mask_ratio)
        
        # return visible_tokens, mask_full
    

    if __name__ == "__main__":
        config = load_config()
        UCF_ROOT_PATH = config['dataset']['root_path']
        UCF_ANNOTATION_PATH = config['dataset']['annot_path']

        try:
            video_mae_transform = transforms.Compose([
            transforms.Resize((224, 224)), # Standard input size for many models
            transforms.Lambda(lambda x: x / 255.0), # Normalize pixel values to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

            mae_processor = VideoMAEClipProcessor(num_frames=16, stride=4, transform = video_mae_transform)
            ucf_dataset = CustomUCF101ForVideoMAE(
                root=UCF_ROOT_PATH,
                annotation_path=UCF_ANNOTATION_PATH,
                frames_per_clip=64, # Needs to be large enough for the processor to sample from
                step_between_clips=100000,
                fold=1,
                train=True,
                video_mae_processor=mae_processor
            )

            data_loader = DataLoader(ucf_dataset, batch_size=1)
            clip, label = next(iter(data_loader))

            model = VideoMAE_mask_one_video()
            visible_tokens, mask_full = model(clip, mask_ratio=0.95)
            print(f"Shape of visible tokens after masking: {visible_tokens.shape}")
            print(f"Shape of the boolean mask: {mask_full.shape}")
            print("\nTest with actual data complete! The model processed the UCF101 batch successfully.")





        
        except FileNotFoundError as e:
            print(f"\nError: Could not find dataset files. Please check the paths.")
            print(f"UCF_ROOT_PATH: {UCF_ROOT_PATH}")
            print(f"UCF_ANNOTATION_PATH: {UCF_ANNOTATION_PATH}")
            print(f"Original error: {e}")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")





     
        



