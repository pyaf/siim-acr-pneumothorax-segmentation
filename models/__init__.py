#from .unet_model import *
from .pretrained import *



def get_model(model_name, num_classes):
    if model_name == "UNet":
        return UNet()
        #return UNet(num_channels, num_classes)


