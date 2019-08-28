import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
#from utils import *
import albumentations
from albumentations import torch as AT


class SIIMDataset(Dataset):
    def __init__(self, df, data_folder, size, mean, std, phase):
        self.df = df
        self.root = os.path.join(data_folder, "npy_files/npy_train_256")
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)

        self.fnames = self.df['ImageId'].tolist()
        self.labels = self.df['has_mask'].values.astype(np.int32) # [12]

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        image_path = os.path.join(self.root, image_id + '.npy')
        img = np.load(image_path)
        img = np.repeat(img, 3, axis=-1)
        #print(img.shape)

        augmented = self.transforms(image=img)#, mask=mask)
        img = augmented['image']# / 255.0

        target = {}
        target["labels"] = self.labels[idx]
        target["image_id"] = image_id
        return img, target

    def __len__(self):
        #return 20
        return len(self.fnames)



def get_transforms(phase, size, mean, std):
    list_transforms = [
        # albumentations.Resize(size, size) # now doing this in __getitem__()
    ]
    if phase == "train":
        list_transforms.extend(
            [
                #albumentations.Transpose(p=0.5),
                #albumentations.Flip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=20,
                    p=0.7,
                    border_mode=1 #cv2.BORDER_CONSTANT
                ),
                albumentations.OneOf(
                    [
                        albumentations.CLAHE(clip_limit=2),
                        albumentations.IAASharpen(),
                        albumentations.IAAEmboss(),
                        albumentations.RandomBrightnessContrast(),
                        albumentations.JpegCompression(),
                        albumentations.Blur(),
                        albumentations.GaussNoise(),
                    ],
                    p=0.5,
                ),
            ]
        )

    list_transforms.extend(
        [

            albumentations.Resize(size, size),
            albumentations.Normalize(mean=mean, std=std, p=1),
            AT.ToTensor(normalize=None), # [6]
        ]
    )
    return albumentations.Compose(list_transforms)

def get_sampler(df, class_weights=None):
    if class_weights is None:
        labels, label_counts = np.unique(
            df["diagnosis"].values, return_counts=True
        )  # [2]
        # class_weights = max(label_counts) / label_counts # higher count, lower weight
        # class_weights = class_weights / sum(class_weights)
        class_weights = [1, 1, 1, 1, 1]
    print("weights", class_weights)
    dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler

def resampled(df):

    ''' resample `total` data points from old data, following the dist of org data '''
    def sample(obj): # [5]
        return obj.sample(n=count_dict[obj.name], replace=False)

    count_dict = {
        0: 10000,
        2: 5292,
        1: 2443,
        3: 873,
        4: 708
    } # notice the order of keys

    sampled_df = train_old.groupby('diagnosis').apply(sample).reset_index(drop=True)

    return sampled_df


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
    df = df.drop_duplicates('ImageId')
    df_with_mask = df.query('has_mask == 1')
    #df = df_with_mask.copy()
    df_without_mask = df.query('has_mask==0')
    df_wom_sampled = df_without_mask.sample(len(df_with_mask)+3000, random_state=69)
    df = pd.concat([df_with_mask, df_wom_sampled])

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    #print(df.shape)
    image_dataset = SIIMDataset(df, images_folder, size, mean, std, phase)
    #datasampler = get_sampler(df, [1, 1])
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
    data_folder = "../data"
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
        labels = targets['labels']
        #pdb.set_trace()
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
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
"""
