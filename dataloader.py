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
#from utils import to_multi_label
import albumentations
from albumentations import torch as AT
from mask_functions import *


class SIIMDataset(Dataset):
    def __init__(self, df, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)

        self.df = self.df.drop_duplicates('ImageId')
        self.fnames = self.df['ImageId'].tolist()
        self.labels = (self.df['EncodedPixels'] != '-1').values.astype(np.int32)

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        image_path = os.path.join(self.root, "npy_train_256",  image_id + '.npy')
        mask_path = os.path.join(self.root, "npy_masks_256", image_id + '.npy')

        #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #img = np.expand_dims(img, -1) # [10]
        img = np.load(image_path)
        #mask_idx = np.load(mask_path)
        #mask = np.zeros([1024, 1024])
        #mask[mask_idx[:, 0], mask_idx[:, 1]] = 1
        mask = np.load(mask_path)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image'] / 255.0
        mask = augmented['mask']

        target = {}
        target["labels"] = self.labels[idx]
        target["image_id"] = image_id
        target["masks"] = mask
        return img, target

    def __len__(self):
        #return 100
        return len(self.fnames)


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

            #albumentations.Normalize(mean=mean, std=std, p=1),
            albumentations.Resize(size, size),
            AT.ToTensor(),  # [6]
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
    mean=None,
    std=None,
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
    num_workers = 12
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
    #images_folder = os.path.join(root, data_folder, "train_png/")  #
    df_path = os.path.join(root, data_folder, train_df_name)  #

    dataloader = provider(
        fold,
        total_folds,
        data_folder,
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
    #pdb.set_trace()

    print('Unique label count:', np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))
    print('fnames unique count:', np.unique(list(fnames_dict.values()), return_counts=True))
    pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2]: masks are not normalized in albumentation's Normalize function
[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
[10]: albumentation's ToTensor supports (w, h) images, no grayscale, so (w, h, 1). IMP: It doesn't give any warning, returns transposed image (weird, yeah)
"""
