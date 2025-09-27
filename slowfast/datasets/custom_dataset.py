import os
import torch
import torch.utils.data
import torchvision.io as io
import torchvision.transforms as T
from torchvision.io import read_video

# from slowfast.datasets import utils as utils
# from slowfast.datasets.build import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
# class Custom(torch.utils.data.Dataset):
#     def __init__(self, cfg, split):
#         self.cfg = cfg
#         self.split = split
#         self.data_path = os.path.join(cfg.DATA.PATH_PREFIX, split)

#         # Assume structure: data_path/class_x/video.mp4
#         self.samples = []
#         classes = sorted(os.listdir(self.data_path))
#         self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

#         for cls in classes:
#             cls_path = os.path.join(self.data_path, cls)
#             if not os.path.isdir(cls_path):  # ✅ skip files
#                 continue
#             for fname in os.listdir(cls_path):
#                 if fname.endswith(".mp4"):
#                     self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

#         # transforms
#         self.transform = T.Compose([
#             T.Resize((cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE)),
#             T.ConvertImageDtype(torch.float32),
#         ])

#     def __getitem__(self, index):
#         video_path, label = self.samples[index]
#         frames, _, _ = io.read_video(video_path, pts_unit="sec")

#         # simple transform
#         frames = self.transform(frames)

#         return frames, label, index, 0  # keep 4 outputs for compatibility

#     def __len__(self):
#         return len(self.samples)




from slowfast.datasets import build_dataset
from slowfast.datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Custom(torch.utils.data.Dataset):
    def __init__(self, cfg, split="train", num_frames=16, frame_rate=1):
        self.cfg = cfg
        self.split = split
        self.num_frames = num_frames
        self.frame_rate = frame_rate
    
        # Path to the split (train/val/test)
        self.data_path = os.path.join(cfg.DATA.PATH_PREFIX, split)
        assert os.path.exists(self.data_path), f"Path not found: {self.data_path}"
    
        # Collect class names
        self.classes = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
        # Collect video paths + labels
        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(self.data_path, cls)
            for vid in os.listdir(cls_folder):
                if not vid.endswith((".mp4", ".avi", ".mov")):
                    continue
                video_path = os.path.join(cls_folder, vid)
                label = self.class_to_idx[cls]
                self.samples.append((video_path, label))
    
        print(f"Loaded {len(self.samples)} videos from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]

        # read_video returns [T, H, W, C], convert to [T, C, H, W]
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)

        # pad / sample frames
        num_total_frames = video.shape[0]
        if num_total_frames < self.num_frames * self.frame_rate:
            pad = self.num_frames * self.frame_rate - num_total_frames
            pad_frames = video[-1:].repeat(pad, 1, 1, 1)
            video = torch.cat([video, pad_frames], dim=0)

        start_idx = random.randint(0, max(0, video.shape[0] - self.num_frames * self.frame_rate))
        indices = start_idx + torch.arange(0, self.num_frames * self.frame_rate, self.frame_rate)
        clip = video[indices]

        clip = clip.float() / 255.0
        clip = torch.nn.functional.interpolate(clip, size=(224, 224), mode='bilinear', align_corners=False)

        meta = {"video_path": video_path}
        return clip, label, index, meta



# from torch.utils.data import Dataset
# import torch
# import os
# import random
# from torchvision.io import read_video

# from slowfast.datasets import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
# class Custom(torch.utils.data.Dataset):
#     def __init__(self, cfg, split="train", num_frames=16, frame_rate=1, num_clips=3):
#         self.cfg = cfg
#         self.split = split
#         self.num_frames = num_frames
#         self.frame_rate = frame_rate
#         self.num_clips = num_clips  # Number of clips to sample per video per __getitem__

#         self.data_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, split)
#         assert os.path.exists(self.data_path), f"Path not found: {self.data_path}"

#         self.classes = sorted(os.listdir(self.data_path))
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

#         self.samples = []
#         for cls_name in self.classes:
#             cls_folder = os.path.join(self.data_path, cls_name)
#             for vid in os.listdir(cls_folder):
#                 if vid.endswith(".avi") or vid.endswith(".mp4"):
#                     self.samples.append((os.path.join(cls_folder, vid), self.class_to_idx[cls_name]))

#         print(f"Loaded {len(self.samples)} videos from {len(self.classes)} classes")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         video_path, label = self.samples[index]
#         video, _, _ = read_video(video_path, pts_unit='sec')  # [T, H, W, C]
#         video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

#         # Pad if video is shorter than required
#         required_frames = self.num_frames * self.frame_rate
#         if video.shape[0] < required_frames:
#             pad = required_frames - video.shape[0]
#             video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

#         clips = []
#         for _ in range(self.num_clips):
#             # Random start for each clip
#             start_idx = random.randint(0, video.shape[0] - required_frames)
#             indices = start_idx + torch.arange(0, required_frames, self.frame_rate)
#             clip = video[indices]
#             clip = clip.float() / 255.0
#             clip = torch.nn.functional.interpolate(clip, size=(224, 224), mode='bilinear', align_corners=False)
#             clips.append(clip)

#         # Return as [num_clips, num_frames, C, H, W]
#         clips = torch.stack(clips, dim=0)
#         meta = {"video_path": video_path}

#         return clips, label, index, meta
