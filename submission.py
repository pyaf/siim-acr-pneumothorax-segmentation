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
from augmentations import *
from utils import *
from image_utils import *
from mask_functions import *

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = ArgumentParser()
    #parser.add_argument("-c", "--ckpt_path",
    #                    dest="ckpt_path", help="Checkpoint to use")
    parser.add_argument(
        "-f",
        "--file",
        dest="filepath",
        help="experiment config file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        dest="epoch",
        help="Epoch to use ckpt of",
    )  # usage: -e 10

    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="test",
    )
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, cfg, tta=4):
        self.root = root
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        size = cfg["size"]
        mean = eval(cfg["mean"])
        std = eval(cfg["std"])
        self.TTA = get_tta()
        self.transform = albumentations.Compose(
            [
                albumentations.Resize(size, size),
                albumentations.Normalize(mean=mean, std=std, p=1),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".npy")
        img = np.load(path)
        img = np.repeat(img, 3, axis=-1)
        images = [self.transform(image=img)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def post_process(probability, threshold, min_size):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


if __name__ == "__main__":
    """
    uses given ckpt to predict on test data and save sigmoid outputs as npy file.
    """
    parser = get_parser()
    args = parser.parse_args()
    epoch = args.epoch
    cfg = load_cfg(args)
    predict_on = args.predict_on

    size = cfg['size']
    num_workers = cfg['num_workers']
    batch_size = cfg['batch_size']['test']
    if predict_on == "test":
        sample_submission_path = cfg["sample_submission"]
        root = 'data/npy_files/npy_test_stage2/'
    else:
        sample_submission_path = cfg["df_path"]
        root = os.path.join(cfg['data_folder'], 'npy_train_512')

    folder = os.path.splitext(os.path.basename(args.filepath))[0]
    model_folder_path = os.path.join('weights', folder)
    npy_folder = os.path.join(model_folder_path, f"{predict_on}_npy/{size}")
    if epoch:
        ckpt_path = os.path.join(model_folder_path, f'ckpt{epoch}.pth')
        sub_path = os.path.join(npy_folder, f"{predict_on}_ckpt{epoch}.csv")
    else:
        ckpt_path = os.path.join(model_folder_path, f'model.pth')
        sub_path = os.path.join(npy_folder, f"{predict_on}_model.csv")


    mkdir(npy_folder)
    npy_path = sub_path.replace(".csv", ".npy")
    tta = 0  # number of augs in tta

    #root = f"data/stage2/{predict_on}_png/"
    save_npy = False
    save_rle = True
    min_size = 2000
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    #setup(use_cuda)
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, cfg, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(cfg)
    model.to(device)
    model.eval()

    print(f"Using {ckpt_path}")
    print(f"Using {sample_submission_path}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"Using tta: {tta}\n")

    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    epoch = state['epoch']
    print(f'Epoch: {epoch}')
    #exit()
    #best_th = state["best_threshold"]
    best_th = 0.5
    print('best_threshold', best_th)
    num_batches = len(testset)
    predictions = []
    encoded_pixels = []
    npy_count = 0
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
            preds = preds.detach().cpu().numpy()[:, 0, :, :]  # [1]
            if save_npy:
                predictions.extend(preds.tolist())
            if save_rle:
                for probability in preds:
                    if probability.shape != (1024, 1024):
                        probability = cv2.resize(probability, (1024, 1024),
                                interpolation=cv2.INTER_LINEAR)
                    predict, num_predict = post_process(probability, best_th, min_size)
                    if num_predict == 0:
                        encoded_pixels.append('-1')
                    else:
                        r = run_length_encode(predict)
                        encoded_pixels.append(r)

        if save_npy:
            if (i+1) % (num_batches//10) == 0:
                print('saving pred npy')
                np.save(npy_path % npy_count, predictions) # raw preds
                npy_count += 1
                del predictions
                predictions = []

    if save_npy:
        np.save(npy_path % npy_count, predictions) # raw preds
        print("Done!")

    if save_rle:
        df['EncodedPixels'] = encoded_pixels
        df.to_csv(sub_path, columns=['ImageId', 'EncodedPixels'], index=False)

