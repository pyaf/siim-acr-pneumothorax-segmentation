import os
import pdb
import cv2
import time
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
from sklearn.metrics import cohen_kappa_score, accuracy_score
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


def predict(X, coef):
    X_p = np.copy(X)
    return (X_p > coef).astype('int')


def compute_score_inv(thresholds, predictions, targets):
    predictions = predict(predictions, thresholds)
    score = accuracy_score(targets, predictions)
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.save_folder = os.path.join(save_folder, "logs")
        self.best_thresholds = 0.5
        self.base_thresholds = 0.5

    def update(self, targets, outputs):
        '''targets, outputs are detached CUDA tensors'''
        # get multi-label to single label
        #targets = torch.sum(targets, 1) - 1 # no multilabel target in regression
        targets = targets.type(torch.LongTensor)
        outputs = torch.sigmoid(outputs).flatten() # [n, 1] -> [n]
        # outputs = torch.sum((outputs > 0.5), 1) - 1

        #pdb.set_trace()
        self.targets.extend(targets.tolist())
        self.predictions.extend(outputs.tolist())
        # self.predictions.extend(torch.argmax(outputs, dim=1).tolist()) #[2]

    def get_best_thresholds(self):
        '''Epoch over, let's get targets in np array [6]'''
        self.targets = np.array(self.targets)

        if self.phase == "train":
            return self.best_thresholds

        """Used in the val phase of iteration, see [4]"""
        self.predictions = np.array(self.predictions)
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            self.best_thresholds,
            args=(self.predictions, self.targets),
            method="nelder-mead",
        )
        self.best_thresholds = simplex["x"][0]
        print("Best thresholds: %s" % self.best_thresholds)
        return self.best_thresholds

    def get_cm(self):
        #pdb.set_trace()
        base_preds = predict(self.predictions, self.base_thresholds)
        cm = ConfusionMatrix(self.targets, base_preds)

        best_preds = predict(self.predictions, self.best_thresholds)
        best_acc = accuracy_score(self.targets, best_preds)
        return cm, best_acc


def print_time(log, start, string):
    diff = time.time() - start
    log(string + ": %02d:%02d" % (diff // 60, diff % 60))


def adjust_lr(lr, optimizer):
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


def epoch_log(log, tb, phase, epoch, epoch_loss, meter, start):
    #diff = time.time() - start
    cm, best_acc = meter.get_cm()
    acc = cm.overall_stat["Overall ACC"]
    tpr = cm.overall_stat["TPR Macro"] #[7]
    ppv = cm.overall_stat["PPV Macro"]
    cls_tpr = cm.class_stat['TPR']
    cls_ppv = cm.class_stat['PPV']
    tpr = 0 if tpr is "None" else tpr  # [8]
    ppv = 0 if ppv is "None" else ppv
    #pdb.set_trace()
    print()
    log(
        "%s %d |  loss: %0.4f | best_acc: %0.4f | ACC: %0.4f | TPR: %0.4f | PPV: %0.4f \n"
        % (phase, epoch, epoch_loss, best_acc, acc, tpr, ppv)
    )
    try:
        cls_tpr = {x: "%0.4f" % y for x, y in cls_tpr.items()}
        cls_ppv = {x: "%0.4f" % y for x, y in cls_ppv.items()}
    except:
        pass

    log('Class TPR: %s' % cls_tpr)
    log('Class PPV: %s' % cls_ppv)
    log(cm.print_normalized_matrix())
    #log("Time taken for %s phase: %02d:%02d \n", phase, diff // 60, diff % 60)

    # tensorboard
    logger = tb[phase]
    logger.log_value("loss", epoch_loss, epoch)
    logger.log_value("ACC", acc, epoch)
    logger.log_value("Best_ACC", best_acc, epoch)
    logger.log_value("TPR", tpr, epoch)
    logger.log_value("PPV", ppv, epoch)

    # save pycm confusion
    obj_path = os.path.join(meter.save_folder, f"cm{phase}_{epoch}")
    cm.save_obj(obj_path, save_stat=True, save_vector=False)

    return best_acc


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_hyperparameters(trainer, remark):
    hp_file = os.path.join(trainer.save_folder, "parameters.txt")
    time_now = datetime.now()
    augmentations = trainer.dataloaders['train'].dataset.transforms.transforms
    # pdb.set_trace()
    string_to_write =  \
        f"Time: {time_now}\n" + \
        f"model_name: {trainer.model_name}\n" + \
        f"train_df_name: {trainer.train_df_name}\n" + \
        f"resume: {trainer.resume}\n" + \
        f"pretrained: {trainer.pretrained}\n" + \
        f"pretrained_path: {trainer.pretrained_path}\n" + \
        f"folder: {trainer.folder}\n" + \
        f"fold: {trainer.fold}\n" + \
        f"total_folds: {trainer.total_folds}\n" + \
        f"num_samples: {trainer.num_samples}\n" + \
        f"sampling class weights: {trainer.class_weights}\n" + \
        f"size: {trainer.size}\n" + \
        f"top_lr: {trainer.top_lr}\n" + \
        f"base_lr: {trainer.base_lr}\n" + \
        f"num_workers: {trainer.num_workers}\n" + \
        f"batchsize: {trainer.batch_size}\n" + \
        f"momentum: {trainer.momentum}\n" + \
        f"mean: {trainer.mean}\n" + \
        f"std: {trainer.std}\n" + \
        f"start_epoch: {trainer.start_epoch}\n" + \
        f"batchsize: {trainer.batch_size}\n" + \
        f"augmentations: {augmentations}\n" + \
        f"criterion: {trainer.criterion}\n" + \
        f"optimizer: {trainer.optimizer}\n" + \
        f"remark: {remark}\n"

    with open(hp_file, "a") as f:
        f.write(string_to_write)
    print(string_to_write)


def seed_pytorch(seed=69):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
"""
