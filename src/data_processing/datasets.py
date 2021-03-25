import os

import pandas as pd
from torchvision.datasets import ImageFolder


class Imagenette(ImageFolder):
    def __init__(self, root, subset='train', csv="noisy_imagenette.csv", **kwargs):
        data_path = os.path.join(root, subset)
        super().__init__(data_path, **kwargs)
        csv_path = os.path.join(root, csv)
        ds = pd.read_csv(csv_path)
        self.classes = list(set(ds.noisy_labels_0))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.imgs = self.get_imgs(root, ds)

    def get_imgs(self, root, ds):
        return [(os.path.join(root, path), self.class_to_idx[target])
                for path, target in zip(ds.path, ds.noisy_labels_0)]
