import gc
import json
import torch
import os
import io
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, random_split
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data.dataloader import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class lrScheduler:
    def __init__(self, optimizer, patience=3, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True)

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class CustomDataset(Dataset):
    def __init__(self, data_base, batches=10):
        self.inputs = [i for i in self.z.namelist() if i.startswith(f'data/{root_dir}')]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x_im = Image.open(io.BytesIO(self.z.read(self.inputs[idx])))
        y = self.targets[idx]

        if self.type_targets == 'img':
            y_im = Image.open(io.BytesIO(self.z.read(y)))
            if self.transform:
                x, y = self.transform(x_im, y_im)
            x, y = CustomDataset.using_transforms(x_im, y_im)
        elif self.transform:
            x = CustomDataset.using_transforms(self.transform(x_im))
        else:
            x = CustomDataset.using_transforms(x_im)

        if self.masks and self.masks_model:
            raise ValueError
        elif self.masks or self.masks_model:
            if self.masks:
                mask_im = Image.open(io.BytesIO(self.z.read(self.masks[idx])))
            elif self.masks_model and self.round:
                mask_im = transforms.functional.to_pil_image(torch.round(self.masks_model(x[None].to(self.device))[0]))
            elif self.masks_model:
                mask_im = transforms.functional.to_pil_image(v2.transforms.to_dtype(self.masks_model(x[None].to(self.device))[0], torch.float32, scale=True))

            if self.transform:
                x_im, mask_im = self.transform(x_im, mask_im)

            if self.resize:
                size = x_im.size[0]
                opencvImage = cv2.cvtColor(np.array(mask_im), cv2.COLOR_RGB2BGR)
                seg_value = [255, 255, 255]
                if opencvImage is not None:
                    np_seg = np.array(opencvImage)
                    segmentation = np.where(np_seg == seg_value)
                    # Bounding Box
                    bbox = 0, 0, 0, 0
                    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                        x_min = int(np.min(segmentation[1]))
                        x_max = int(np.max(segmentation[1]))
                        y_min = int(np.min(segmentation[0]))
                        y_max = int(np.max(segmentation[0]))
                        bbox = (x_min, y_min, x_max, y_max)

                if self.resize == 1:
                    side = max(x_max - x_min, y_max - y_min)
                    black_square1 = Image.new('RGB', (side, side))
                    black_square2 = Image.new('RGB', (side, side))
                    black_square1.paste(x_im.crop(bbox), ((side - x_max + x_min) // 2, (side - y_max + y_min) // 2))
                    black_square2.paste(mask_im.crop(bbox), ((side - x_max + x_min) // 2, (side - y_max + y_min) // 2))

                    x, mask = CustomDataset.using_transforms(black_square1.resize((size, size)), black_square2.resize((size, size)))
                elif self.resize == 2:
                    x, mask = CustomDataset.using_transforms(x_im.resize((size, size), box=bbox), mask_im.resize((size, size), box=bbox))
            else:
                x, mask = CustomDataset.using_transforms(x_im, mask_im)

            if self.one_layer_mask:
                x = x * mask
            else:
                x = torch.cat((x, x * mask, mask), dim=-3)
        return x, y