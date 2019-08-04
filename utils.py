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
import torch.backends.cudnn as cudnn
from tensorboard_logger import log_value, log_images
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix
from extras import *


def setup(use_cuda):
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)

    with torch.no_grad():

        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def compute_score_inv(threshold, predictions, targets):
    #pdb.set_trace()
    predictions = predict(predictions, threshold)
    #score = compute_iou_batch(predictions, targets, classes=[1])
    score = compute_dice(predictions, targets)
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.probabilities = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.size = 256
        self.save_folder = os.path.join(save_folder, "logs")
        self.base_threshold = 0.7
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        """targets, outputs are detached CUDA tensors"""
        #pdb.set_trace()
        #outputs = outputs.cpu().numpy()
        #targets = targets.cpu().numpy()
        #pdb.set_trace()
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

        if self.phase == "val": # [10]
            for prob, targ in zip(probs,targets):
                prob = cv2.resize(prob[0].numpy(), (self.size, self.size))
                targ = cv2.resize(targ[0].numpy(), (self.size, self.size))
                self.probabilities.append(prob)
                self.targets.append(targ)

    def get_best_threshold(self):
        '''[4], [6]'''
        if self.phase == "train": return None
        self.targets = np.array(self.targets)
        self.probabilities = np.array(self.probabilities)
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            self.base_threshold,
            args=(self.probabilities, self.targets),
            method="nelder-mead",
        )
        self.best_threshold = simplex["x"][0]
        print("Best threshold: %s" % self.best_threshold)
        return self.best_threshold

    def get_best_dice(self):
        best_preds = predict(self.probabilities, self.best_threshold)
        best_dice = compute_dice(best_preds, self.targets)
        return best_dice

    def get_metrics(self):
        #pdb.set_trace()
        dices = self.get_dices()
        iou = np.nanmean(self.iou_scores)
        return dices, iou

    def get_dices(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        return dice, dice_neg, dice_pos


def adjust_lr(lr, optimizer):
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


def epoch_log(opt, log, tb, phase, epoch, epoch_loss, meter, start):
    logger = tb[phase]
    if phase == "val":
        best_dice = meter.get_best_dice()
        log("best_dice: %0.04f" % best_dice)
        logger.log_value("best_dice", best_dice, epoch)

    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    lr = opt.param_groups[-1]["lr"]

    log("lr: %f | IoU: %0.4f" % (lr, iou))
    log("dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (dice, dice_neg, dice_pos))

    logger.log_value("lr", lr, epoch)
    logger.log_value("loss", epoch_loss, epoch)
    logger.log_value("IoU", iou, epoch)
    logger.log_value("dice", dice, epoch)
    logger.log_value("dice_neg", dice_neg, epoch)
    logger.log_value("dice_pos", dice_pos, epoch)

    return None

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
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(compute_ious(pred, label, classes))
    iou = np.nanmean(ious)
    return iou


def compute_dice(preds, target):
    eps = 1 # dice is 1 when both are empty.
    outputs = np.copy(preds) # IMP
    inter = np.sum(outputs * target)
    union = np.sum(outputs) + np.sum(target) + eps
    t = (2 * inter + eps) / union
    return t


def collate_fn(batch):
    pdb.set_trace()
    images, targets = zip(*batch)
    images = torch.Tensor(images)
    #targets is a tuple of dicts
    return images, targets



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
