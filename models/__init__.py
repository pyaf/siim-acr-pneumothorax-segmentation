from .unet_model import *



def get_model(model_name, num_channels=1, num_classes=1):
    if model_name == "UNet":
        return UNet(num_channels, num_classes)


