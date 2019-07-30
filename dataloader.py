import os
import cv2
import pdb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import to_multi_label
import albumentations
from albumentations import torch as AT
from mask_functions import *


class SIIMDataset(Dataset):
    def __init__(self, df, img_dir, size, mean, std, phase):
        self.df = df
        self.image_dir = img_dir
        self.height = size
        self.width = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.image_info = defaultdict(dict)

        self.transforms = get_transforms(phase, size, mean, std)
        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id + '.png')
            if os.path.exists(image_path) and int(row["has_mask"]):
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row["EncodedPixels"].strip()
                counter += 1
        print(f"counter: {counter}")

    def __getitem__(self, idx):
        info = self.image_info[idx]

        img_path = info["image_path"]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        mask = rle2mask(info['annotations'], width, height)
        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        #mask = np.expand_dims(mask, axis=0)

        #labels = torch.ones((1,), dtype=torch.int64)
        #labels = torch.Tensor([1])

        target = {}
        #target["labels"] = labels
        target["image_id"] = info['image_id']
        img = np.asarray(img)
        mask = np.asarray(mask)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        #masks = torch.as_tensor(mask, dtype=torch.uint8)
        target["masks"] = mask / 255.0 # [2] .type('torch.LongTensor') # int64

        return img, target

    def __len__(self):
        return len(self.image_info)


def get_transforms(phase, size, mean, std):
    list_transforms = [
    ]
    if phase == "train":
        list_transforms.extend(
            [
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=120,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
    list_transforms.extend(
        [

            albumentations.Normalize(mean=mean, std=std, p=1),
            #albumentations.Resize(size, size),
            AT.ToTensor(normalize=None),  # [6]
        ]
    )
    return albumentations.Compose(list_transforms)


def provider(
    fold,
    total_folds,
    images_folder,
    df_path,
    phase,
    size,
    mean,
    std,
    class_weights=None,
    batch_size=8,
    num_workers=4,
    num_samples=4000,
):
    df = pd.read_csv(df_path)
    HOME = os.path.abspath(os.path.dirname(__file__))

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    df = train_df if phase == "train" else val_df

    image_dataset = SIIMDataset(df, images_folder, size, mean, std, phase)
    datasampler = None
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args

    #print(f'len(dataloader): {len(dataloader)}')
    return dataloader


if __name__ == "__main__":
    import time
    start = time.time()
    phase = "train"
    #phase = "val"
    num_workers = 0
    fold = 0
    total_folds = 5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #mean = (0.5, 0.5, 0.5)
    #std = (0.5, 0.5, 0.5)

    size = 256

    root = os.path.dirname(__file__)  # data folder
    data_folder = "data"
    train_df_name = 'train.csv'
    num_samples = None  # 5000
    class_weights = True  # [1, 1, 1, 1, 1]
    batch_size = 16
    images_folder = os.path.join(root, data_folder, "train_png/")  #
    df_path = os.path.join(root, data_folder, train_df_name)  #

    dataloader = provider(
        fold,
        total_folds,
        images_folder,
        df_path,
        phase,
        size,
        mean,
        std,
        class_weights=class_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        images, targets = batch
        masks = targets['masks']
        labels = targets['labels']
        #pdb.set_trace()
        for fname in targets['image_id']:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, masks.shape, labels.shape)
        total_labels.extend(labels.tolist())
        # pdb.set_trace()
    #print(np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))

    print(np.unique(list(fnames_dict.values()), return_counts=True))
    pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2]: masks are not normalized in albumentation's Normalize function
[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
"""
