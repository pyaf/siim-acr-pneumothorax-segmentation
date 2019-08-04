# Logs for the competition

# ADPOS diabetic retina

# Models on training:

https://github.com/qubvel/segmentation_models

* 5e-4 is too high for lr *


### 1 Aug

The UNet model which I was using so far didn't had BN layers. Now, using a new model looks like its working,

*BUG* len(dataloader) returns total batches not total images.
BCEWithLogitsLoss has reduction="mean" the loss returned is mean of all elements in outputs, here total number of pixels

Study unet original architecture, compare it with your model.

Trainer.size was set to 224 -_-, i wasn't using albu's resize anyway :D

*_with_logits means don't use sigmoid ouputs, it'll take care of it.
Logits are interpreted to be the unnormalised (or not-yet normalised) predictions (or outputs) of a model.


Model is learning now, there were two issues, the output image from the dataloader was normalized two times, 1 by explicit division by 255.0, 2 by AT.ToTensor.
other issue which was unncessary was sigmoid input to loss with logits.
model had F.sigmoid() at the end -_-
sample_submission scores 0.76 -_- I have been scoring ~0.2 for past 3 days

Can't use tta in this problem
there was a but, wasn't using `base_preds` in compute_dice, instead was using sigmoid output
one more bug: self.predictions were sigmoid outputs, not thresholded ones.


* `18_UNet_f1_dice`: dice metric implemented. training on pos + pos+2k neg images.

read the forum, all of it.

* `28_UNet_f1_eq`: using DiceLoss as criterion, finally, pos dice is above 0.30
model is being trained with lr: 5e-4, retraining with 1e-3
retraining with SGD instead of Adam, NOPE, SGD doesn't work.
retraining with pos + pos+1k neg images, eq sampler.


Trained a classifier: `2-8_efficientnet_b5_f1_` on 1/7 fold, used cktp20 for predictions. ACC: 0.87/0.87, PPV: 0.85/0.85, TPR: 0.85/0.85
val:
Class TPR: {0: '0.9090', 1: '0.7912'}
Class PPV: {0: '0.9078', 1: '0.7935'}

Analysing the classification predictions, got to know that the seg model had ~200 FPs and 165 FNs. After substituting FPs with '-1', got LB dice boost from 0.66 to 0.79

analysing the train preds of seg model, it is not learning anything, random things instead.


### 3 Aug

* `38_UNet_f1_HF`: removed augs of flip and transpose and random scale, only horizontal flip. with criterion.
OMG OMG OMG, something is happening. OMG I'm crying, this was the fucking issue, you don't need 360 deg rotation and random flip even vertical and transpose, fml |_|
dice 0.15 in first epoch, no this wasn't the issue
learn to read the forums \m/
not getting above 0.2 in train dice, retraining with grad accu for bs 32, with BCELossWithLogits, no aug at all.
BCELossWithLogits getting biased towards background class
Nothing worked.

Used UNet with resnet encoder pretrained on imagenet => boom 0.81 LB, though train/val was 0.57/0.49 on pos only and 0.65 on pos + (pos+3k) neg images

with np.load 256 sized image, dataloader takes 2:07 for iteration over dataset
with Image.open on train_img convert RGB, to np arr, resize in albumentations, it takes, hyperthreads not fully utilized, 4:42
with cv2.imread on train_img, cv2.convert, resize albumentations, it takes: 4:28, hyperthreads not utilized.


saving images in 512 size

dataloader takes 4:08 without AT.ToTensor and with manual torch.Tensor in __getitem__()
3:05 with it.
eq weighted datasampler doesn't let all the images out of dataloader in one epoch, it's stochastic many images pass through more than once.
removing datasampler increases time to 4:22


* `38_UNet_f1_512` training on 512 npy images, with equal pos and neg images, with aug horizontalflip, rotate(10),
IoU is buggy. loss is fluctuating, will train with 4e-5 next time.
ckpt 23
Val optimized th: 0.402, LB: 0.8385

### 4 Aug

* `48_UNet_f1_ft1024`: finetuning the ckpt23 of previous model on 1024 sized images., batch size =2,
I'm still using equal pos and neg images





OLD kaggle competition:
https://www.kaggle.com/iafoss/unet34-dice-0-8://www.kaggle.com/iafoss/unet34-dice-0-87
install latest of segmentation_models.pytorch


# Questions and Ideas:

# TODO:

* Img size 512 and 1024?


# Revelations:




# Things to check, just before starting the model training:

* train_df_name
* model_name
* fold and total fold (for val %)
* npy_folder_name for dataloader's __getitem__() function
* are you resampling images?
* self.size, self.top_lr, self.std, self.mean -> insta trained weights used so be careful
* self.ep2unfreeze
*



# Observations:

# NOTES:

# Files informations:

* data/train_png: png version of dicom images
* data/npy_masks: contains indices where masks are 1
* data/npy_train: npy files of `train_png`
* data/npy_train_256: same as above, cv2.resized to 256
* data/npy_masks_256: masks cv2.resized to 256, new value points generated in between 0 and 1, so thresholded at 0.5, not indices


# remarks shortcut/keywords

# Experts speak


