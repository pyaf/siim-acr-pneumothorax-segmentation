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
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix

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


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def compute_score_inv(threshold, predictions, targets):
    #pdb.set_trace()
    predictions = predict(predictions, threshold)
    #score = compute_iou_batch(predictions, targets, classes=[1])
    score = compute_dice(predictions, targets)
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.save_folder = os.path.join(save_folder, "logs")
        self.base_threshold = 0.7
        self.best_threshold = 0.7
        self.base_dice_scores = []

    def update(self, targets, outputs):
        """targets, outputs are detached CUDA tensors"""
        #pdb.set_trace()
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        base_preds = predict(outputs, self.base_threshold)
        dice = compute_dice(outputs, targets)
        self.base_dice_scores.append(dice)

        if self.phase == "val": # [10]
            self.targets.extend(targets)
            self.predictions.extend(outputs)

    def get_best_threshold(self):
        if self.phase == "train":
            return self.base_threshold

        """Used in the val phase of iteration, see [4],
        Epoch over, let's get targets in np array [6]"""
        self.targets = np.array(self.targets)
        self.predictions = np.array(self.predictions)
        #pdb.set_trace()
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            self.base_threshold,
            args=(self.predictions, self.targets),
            method="nelder-mead",
        )
        self.best_threshold = simplex["x"][0]
        print("Best threshold: %s" % self.best_threshold)
        return self.best_threshold

    def get_metrics(self):
        #pdb.set_trace()
        base_dice = np.mean(self.base_dice_scores)
        best_dice = base_dice
        if self.phase == "val":
            best_preds = predict(self.predictions, self.best_threshold)
            best_dice = compute_dice(best_preds, self.targets)
        return best_dice, base_dice


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


def adjust_lr(lr, optimizer):
    """ Update the lr of base model
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


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


def epoch_log(log, tb, phase, epoch, epoch_loss, meter, start):
    diff = time.time() - start
    best_dice, base_dice = meter.get_metrics()

    log("best/base dice: %0.4f/%0.4f"
            % (best_dice, base_dice))
    #log("Time taken for %s phase: %02d:%02d", phase, diff // 60, diff % 60)

    # tensorboard
    logger = tb[phase]
    logger.log_value("loss", epoch_loss, epoch)
    logger.log_value("best_dice", best_dice, epoch)
    logger.log_value("base_dice", base_dice, epoch)

    return None

def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''
    `classes` is a list of class labels, ignore background class i.e., 0
    example classes=[1, 2]
    '''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    ious = []
    preds = np.copy(outputs) # copy is imp
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def compute_dice(preds, target):
    eps = 0.0001
    outputs = np.copy(preds) # IMP
    inter = np.sum(outputs * target)
    union = np.sum(outputs) + np.sum(target) + eps
    t = (2 * inter + eps) / union
    return t


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


"""Footnotes:

[1]: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

[2]: Used in cross-entropy loss, one-hot to single label

[3]: # argmax returns earliest/first index of the maximum value along the given axis
 get_preds ka ye hai ki agar kisi output me zero nahi / sare one hain to 5 nahi to jis index par pehli baar zero aya wahi lena hai, example:
[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
-> [4, 1, 0, 0]
baki clip karna hai (0, 4) me, we can get -1 for cases with all zeros.

[4]: get_best_threshold is used in the validation phase, during each phase (train/val) outputs and targets are accumulated. At the end of train phase a threshold of 0.5 is used for
generating the final predictions and henceforth for the computation of different metrics.
Now for the validation phase, best_threshold function is used to compute the optimum threshold so that the qwk is minimum and that threshold is used to compute the metrics.

It can be argued ki why are we using 0.5 for train, then, well we used 0.5 for both train/val so far, so if we are computing this val set best threshold, then not only it can be used to best evaluate the model on val set, it can also be used during the test time prediction as it is being saved with each ckpt.pth

[5]: np.array because it's a list and gets converted to np.array in get_best_threshold function only which is called in val phase and not training phase

[6]: It's important to keep these two in np array, else ConfusionMatrix takes targets as strings. -_-

[7]: macro mean average of all the classes. Micro is batch average or sth.

[8]: sometimes initial values may come as "None" (str)

[9]: I'm using base th for train phase, so base_qwk and best_qwk are same for train phase, helps in comparing the base_qwk and best_qwk of val phase with the train one, didn't find a way to plot base_qwk of train with best and base of val on a single plot.

[10]: gives mem overflow for storing all predictions in training set.
"""
