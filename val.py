import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from PIL import Image
from models import get_model
from utils import *
from image_utils import *
from mask_functions import *

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-c", "--ckpt_path",
                        dest="ckpt_path", help="Checkpoint to use")
    #parser.add_argument(
    #    "-p",
    #    "--predict_on",
    #    dest="predict_on",
    #    help="predict on train or test set, options: test or train",
    #    default="resnext101_32x4d",
    #)
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = albumentations.Compose([])
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, -1)
        image = np.repeat(image, 3, axis=-1)

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_model_name_fold(ckpt_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/ckpt12.pth
    model_folder = ckpt_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2][1:]  # f0 -> 0
    return model_name, int(fold)


if __name__ == "__main__":
    """
    uses given ckpt to predict on val set data and save sigmoid outputs as npy file to be used for optimized thresholds
    """
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    model_name, fold = get_model_name_fold(ckpt_path)

    fold = 1
    total_folds = 5
    df_path = 'data/train.csv'
    df = pd.read_csv(df_path)
    df = df.drop_duplicates('ImageId')
    df_with_mask = df.query('has_mask == 1')
    df_without_mask = df.query('has_mask==0')
    df_wom_sampled = df_without_mask.sample(len(df_with_mask))
    df = pd.concat([df_with_mask, df_wom_sampled])
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    npy_path = ckpt_path.replace(".pth", "val%d.npy")
    tta = 0  # number of augs in tta
    root = f"data/train_png/"
    size = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    use_cuda = True
    num_classes = 1
    num_workers = 8
    batch_size = 16
    device = torch.device("cuda" if use_cuda else "cpu")
    setup(use_cuda)
    testset = DataLoader(
        TestDataset(root, val_df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes)
    model.to(device)
    model.eval()

    print(f"Using {ckpt_path}")
    print(f"Predicting on: val set")
    print(f"Root: {root}")
    print(f"Using tta: {tta}\n")

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            # images.shape [n, 3, 96, 96] where n is num of 1+tta
            for images in batch:
                preds = torch.sigmoid(model(images.to(device)))  # [n, num_classes]
                preds = preds.mean(dim=0).detach().tolist()
                predictions.append(preds)
        else:
            #pdb.set_trace()
            preds = torch.sigmoid(model(batch[:, 0].to(device)))
            preds = preds.detach().tolist()  # [2]

            predictions.extend(preds)
        if (i+1) % 40 == 0:
            print('Saving pred npy')
            np.save(npy_path % ((i+1)//40), predictions) # raw preds
            predictions = []

    np.save(npy_path % 2, predictions) # raw preds
    print("Done!")

