# custom_dataset.py
import os
from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets.video_dataset import VideoDataset

@register_dataset("custom")
class CustomDataset(VideoDataset):
    """
    Example custom dataset for PySlowFast.
    Inherits everything from VideoDataset.
    You just need to point DATA.PATH_TO_DATA_DIR in your config.
    Directory format should be like:
    
    root/
      train/
        class1/
          vid1.mp4
          vid2.mp4
        class2/
          vid3.mp4
      val/
        class1/
        class2/
    """

    def __init__(self, cfg, split):
        # split: "train", "val", or "test"
        super().__init__(cfg, split)

    def __getitem__(self, index):
        """
        Loads the video and label at the given index.
        VideoDataset already handles decoding, sampling, and transforms.
        """
        frames, label, index, time_index, meta = super().__getitem__(index)
        return frames, label, index, time_index, meta
