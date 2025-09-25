import os
import torch
import torch.utils.data
import torchvision.io as io
import torchvision.transforms as T

from slowfast.datasets import utils as utils
from slowfast.datasets.build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.data_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, split)

        # Assume structure: data_path/class_x/video.mp4
        self.samples = []
        classes = sorted(os.listdir(self.data_path))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(self.data_path, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".mp4"):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

        # transforms
        self.transform = T.Compose([
            T.Resize((cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE)),
            T.ConvertImageDtype(torch.float32),
        ])

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        frames, _, _ = io.read_video(video_path, pts_unit="sec")

        # simple transform
        frames = self.transform(frames)

        return frames, label, index, 0  # keep 4 outputs for compatibility

    def __len__(self):
        return len(self.samples)
