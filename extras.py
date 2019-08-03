import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pdb
import cv2
import time
import json
import torch
import random
import scipy
import logging
import traceback
import numpy as np
from datetime import datetime

# from config import HOME
from tensorboard_logger import log_value, log_images
from matplotlib import pyplot as plt

plt.switch_backend("agg")


def logger_init(save_folder):
    mkdir(save_folder)
    logging.basicConfig(
        filename=os.path.join(save_folder, "log.txt"),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    return logger



def plot_ROC(roc, targets, predictions, phase, epoch, folder):
    roc_plot_folder = os.path.join(folder, "ROC_plots")
    mkdir(os.path.join(roc_plot_folder))
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_plot_name = "ROC_%s_%s_%0.4f" % (phase, epoch, roc)
    roc_plot_path = os.path.join(roc_plot_folder, roc_plot_name + ".jpg")
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.legend(["diagonal-line", roc_plot_name])
    fig.savefig(roc_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # see footnote [1]

    plot = cv2.imread(roc_plot_path)
    log_images(roc_plot_name, [plot], epoch)


def print_time(log, start, string):
    diff = time.time() - start
    log(string + ": %02d:%02d" % (diff // 60, diff % 60))



def iter_log(log, phase, epoch, iteration, epoch_size, loss, start):
    diff = time.time() - start
    log(
        "%s epoch: %d (%d/%d) loss: %.4f || %02d:%02d",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss.item(),
        diff // 60,
        diff % 60,
    )


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_hyperparameters(trainer, remark):
    hp_file = os.path.join(trainer.save_folder, "parameters.txt")
    time_now = datetime.now()
    augmentations = trainer.dataloaders["train"].dataset.transforms.transforms
    # pdb.set_trace()
    string_to_write = (
        f"Time: {time_now}\n"
        + f"model_name: {trainer.model_name}\n"
        + f"train_df_name: {trainer.train_df_name}\n"
        #+ f"images_folder: {trainer.images_folder}\n"
        + f"resume: {trainer.resume}\n"
        + f"pretrained: {trainer.pretrained}\n"
        + f"pretrained_path: {trainer.pretrained_path}\n"
        + f"folder: {trainer.folder}\n"
        + f"fold: {trainer.fold}\n"
        + f"total_folds: {trainer.total_folds}\n"
        + f"num_samples: {trainer.num_samples}\n"
        + f"sampling class weights: {trainer.class_weights}\n"
        + f"size: {trainer.size}\n"
        + f"top_lr: {trainer.top_lr}\n"
        + f"base_lr: {trainer.base_lr}\n"
        + f"num_workers: {trainer.num_workers}\n"
        + f"batchsize: {trainer.batch_size}\n"
        + f"momentum: {trainer.momentum}\n"
        + f"mean: {trainer.mean}\n"
        + f"std: {trainer.std}\n"
        + f"start_epoch: {trainer.start_epoch}\n"
        + f"augmentations: {augmentations}\n"
        + f"criterion: {trainer.criterion}\n"
        + f"optimizer: {trainer.optimizer}\n"
        + f"remark: {remark}\n"
    )

    with open(hp_file, "a") as f:
        f.write(string_to_write)
    print(string_to_write)


def seed_pytorch(seed=69):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


