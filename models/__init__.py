from .unet import *



def get_model(model_name, num_classes=1):
    if model_name == "UNet":
        return UNet(num_classes)


