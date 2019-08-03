import segmentation_models_pytorch as smp


def UNet(encoder="resnet34", pretrained="imagenet"):
    model = smp.Unet(encoder, encoder_weights=pretrained, activation=None)
    return model
