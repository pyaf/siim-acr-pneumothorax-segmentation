import segmentation_models_pytorch as smp
from torchvision import models
from torchvision.models.segmentation import *
from torch import nn
from torch.nn import functional as F

def UNet(encoder="resnet34", pretrained="imagenet"):
    model = smp.Unet(encoder, encoder_weights=pretrained, activation=None)
    return model

class Deeplabv3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Deeplabv3, self).__init__()
        dlab = deeplabv3_resnet101(pretrained=True)
        self.backbone = dlab.backbone
        self.classifier = deeplabv3.DeepLabHead(2048, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)['out']
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def get_model(cfg):
    model_name = cfg['model_name']
    if model_name == "UNet":
        encoder = cfg['encoder']
        model = UNet(encoder)

    if model_name == "deeplabv3":
        model = Deeplabv3(cfg['num_classes'])

    return model

