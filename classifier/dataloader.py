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
from augmentations import *
from extras import *
#from mask_functions import *


HOME = os.path.abspath(os.path.dirname(__file__))

class SIIMDataset(Dataset):
    def __init__(self, df, phase, cfg):
        self.df = df
        if phase == "new_val":
            data_folder = cfg['val_data_folder']
        else:
            data_folder = cfg['data_folder']
        self.root = os.path.join(cfg['home'], data_folder)
        self.size = cfg['size']
        self.phase = phase
        self.transforms, self.img_trfms = get_transforms(phase, cfg)

        self.fnames = self.df['ImageId'].tolist()
        self.labels = self.df['has_mask'].values.astype(np.int32) # [12]

    def __getitem__(self, idx):
        image_id = self.fnames[idx]

        #image_path = os.path.join(self.root, "npy_train_512",  image_id + '.npy')
        #img = np.load(image_path)
        #img = np.repeat(img, 3, axis=-1)

        path = os.path.join(self.root, image_id + '.png')
        img = cv2.imread(path)

        augmented = self.transforms(image=img)
        img = augmented['image']# / 255.0
        #extra_augs = self.img_trfms(image=img)
        #img = extra_augs['image']
        target = {}
        target["labels"] = self.labels[idx]
        target["image_id"] = image_id
        return img, target

    def __len__(self):
        #return 100
        return len(self.fnames)


def get_sampler(df, class_weights=[1, 1]):
    dataset_weights = [class_weights[idx] for idx in df['has_mask']]
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler


def provider(phase, cfg):
    if phase in ['train', 'val']:
        df_path = os.path.join(cfg['home'], cfg['df_path'])
        df = pd.read_csv(df_path)
        df = df.drop_duplicates('ImageId')
        df_with_mask = df.query('has_mask == 1')
        #df = df_with_mask.copy()
        df_without_mask = df.query('has_mask==0')
        df_wom_sampled = df_without_mask.sample(len(df_with_mask), random_state=69)
        df = pd.concat([df_with_mask, df_wom_sampled])
        df = df.sample(frac=1, random_state=69)

        fold = cfg['fold']
        total_folds = cfg['total_folds']
        kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
        train_idx, val_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        df = train_df if phase == "train" else val_df

        print(df['has_mask'].value_counts())
    if phase == "new_val":
        df_path = os.path.join(cfg['home'], cfg['val_df_path'])
        df = pd.read_csv(df_path)

    image_dataset = SIIMDataset(df, phase, cfg)
    #datasampler = get_sampler(df, [1, 1])
    datasampler = None

    batch_size = cfg['batch_size'][phase]
    num_workers = cfg['num_workers']

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


def testprovider(cfg):
    HOME = cfg['home']
    df_path = cfg['sample_submission']
    df = pd.read_csv(os.path.join(HOME, df_path))
    phase = cfg['phase']
    if phase == "test":
        df['id_code'] += '.png'
    batch_size = cfg['batch_size']['test']
    num_workers = cfg['num_workers']


    dataloader = DataLoader(
        ImageDataset(df, phase, cfg),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return dataloader




if __name__ == "__main__":
    import time
    start = time.time()
    phase = "train"
    args = get_parser()
    cfg = load_cfg(args)
    cfg["num_workers"] = 8
    cfg["batch_size"]["train"] = 4
    cfg["batch_size"]["val"] = 4

    dataloader = provider(phase, cfg)
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        images, targets = batch
        #masks = targets['masks']
        labels = targets['labels']
        pdb.set_trace()
        for fname in targets['image_id']:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
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
[12], .tolist() gives CUDA initialization error, it needs to be in numpy array with np.int32 dtype to avoid it.
[13]: It is of utmost importance that mask is in float format, mask natively contains 0, 1, if it isn't converted to float32, it'll be divided by 255 in ToTensor()
"""
