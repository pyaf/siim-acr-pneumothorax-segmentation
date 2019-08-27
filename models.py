import segmentation_models_pytorch as smp


def UNet(encoder="resnet34", pretrained="imagenet"):
    model = smp.Unet(encoder, encoder_weights=pretrained, activation=None)
    return model

def get_model(cfg):
    model_name = cfg['model_name']
    if model_name == "UNet":
        encoder = cfg['encoder']
        return UNet(encoder)


