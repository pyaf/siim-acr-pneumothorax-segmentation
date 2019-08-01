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






# Questions and Ideas:

# TODO:

* gotta compute dice score not IOUs.
* multiple masks per image ?



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


