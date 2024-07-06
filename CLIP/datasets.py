import os
import numpy as np
import torch
from typing import Tuple
from torchvision import transforms
from PIL import Image
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data/daichi/MEG") -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = os.path.join(data_dir, "data")
        self.image_data_dir = os.path.join(data_dir, "Images")
        self.num_classes = 1854

        self.X_brainwave = torch.load(os.path.join(self.data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(
            self.data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(self.data_dir, f"{split}_y.pt"))
            with open(os.path.join(self.data_dir, f"{split}_image_paths.txt"), "r") as f:
                self.image_paths = f.read().splitlines()
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.X_brainwave)

    def __getitem__(self, i):
        brainwave = self.X_brainwave[i]
        back_split_dirs = self.image_paths[i].split("/")
        if len(back_split_dirs) == 1:
            split_dirs = self.image_paths[i].split("_")
            if len(split_dirs) == 2:
                image_path = os.path.join(
                    self.image_data_dir, split_dirs[0], self.image_paths[i])
            elif len(split_dirs) == 1:
                image_path = os.path.join(self.image_data_dir, self.image_paths[i])
            elif len(split_dirs) == 3:
                image_path = os.path.join(
                    self.image_data_dir, "_".join(split_dirs[:2]), self.image_paths[i])
        else:
            image_path = os.path.join(self.image_data_dir, self.image_paths[i])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if hasattr(self, "y"):
            label = self.y[i]
            subject_idx = self.subject_idxs[i]
            return brainwave, image, label, subject_idx
        else:
            subject_idx = self.subject_idxs[i]
            return brainwave, image, subject_idx

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
