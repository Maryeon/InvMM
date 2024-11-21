import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FacesHQ(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 flip_p=0.5,
                 exts=['jpg', 'jpeg', 'png', 'tiff']
                 ):
        self.data_root = data_root
        self.image_paths = []
        for path in sorted([p for ext in exts for p in glob.glob(f'{data_root}/*.{ext}')]):
            self.image_paths.append(path)
        self._length = len(self.image_paths)

        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=flip_p)
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        image = np.array(image).astype(np.uint8)

        res = {}
        res["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return res