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


### 27 Aug

I'm back :D

oh gosh!, gotta do so much.

### *💡*

* Save masks with dtype uint8 instead of float32, 4x less space., resaving npy masks in uint8 format. Dayummm, npy_masks_1024: 42GB -> 11GB 🤯
* horizontal flip is detrimental, best lr is 1e-4
* Setting Random states is fucking important.
* Always make sure mask out of dataloader are in [0, 1], ToTensor is buggy, bc uint8 masks are divided by 255.



`278_unetresnet34_f1_test`: just reproducing kernel results.

`278_unetresnet34_f1_test2`: with 0.05, 0.05, 10 shiftscalerotate only

`278_unetresnet34_f1_test3`: with horizontal flip + 0.1, 0.1, 20 shiftscalerotate
val performance is more or less same, train performance is low compared to test2 version.

ckpt19:
dice, dice_pos, dice_neg: 0.69/57, 0.53/48, 0.85/0.73 ------- LB: 0.83

*Why is there such a huge discrepancy in val dice score and LB?*
- because of the sampled dataset I'm training on.


Gotta use NIH chest xray dataset.

https://www.kaggle.com/nih-chest-xrays/data
https://stanfordmlgroup.github.io/competitions/chexpert/
https://physionet.org/content/mimic-cxr/1.0.0/


`188_senet154_f1_test`: model.pth ep28 scores 0.8540


training classifier:

*`288_efficientnet-b5_f1_test`: 5e-5, bad augmentations.
* `288_efficientnet-b5_f1_test2`: 5e-5, decent augs.

*Give home, give path in the dataloader df*


https://github.com/JunMa11/SegLoss

OLD kaggle competition:
https://www.kaggle.com/iafoss/unet34-dice-0-8://www.kaggle.com/iafoss/unet34-dice-0-87
install latest of segmentation_models.pytorch



*Stage2 is active*

Downloaded the new test set to be predicted on. Stage1 train and test set become the train set for stage2. Analysed my test set predictions, the dice is correct, I'm good with neg dice 0.7, but bad with pos dice 0.3, the bin classifier is no good than the seg predictor.

Things to note:
* One thing to note is the gap between train and val loss, Gotta do stronger augmentation.
* Given val loss for resnet34 keeps on decreasing for 1e-4, can try higher lr.
* Stage2 LB is calculated on 1% of the test dataset.


* `298_resnet34_f1_test`:

Ep 25: pos, neg: 0.43/28, 0.80, dice: 0.78/0.72 LB 0.8991, (LB is irrelevant here, only 1% of testdata is used to calculate LB)


* `019_unetden121_f1_test`: densenet121, 512 -> 30 eps, afterwards 1024.











*WHENEVER DEBUGGING MODEL ARCHITECTURES USE BATCH SIZE > 1*

Bang on the loss function.
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-598791

running them back to back with resnet34 encoder.
wlova is 1 on neg dice 0 on pos dice
then comes wbce,
wsd has ~ same perf on pos dice, but mixed loss is better at neg dice.
focal loss alone performs at par with mixed loss.





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



# Observations:

* vim: detele till a char including it df<char>
* getting dense cpu tensor in pin memory error? remove default float tesnor setting.
# NOTES:

# Files informations:

* data/train_png: png version of dicom images
* data/npy_masks: contains indices where masks are 1
* data/npy_train: npy files of `train_png`
* data/npy_train_256: same as above, cv2.resized to 256
* data/npy_masks_256: masks cv2.resized to 256, new value points generated in between 0 and 1, so thresholded at 0.5, not indices


# remarks shortcut/keywords

# Experts speak


