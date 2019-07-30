import pdb
import torch
from torch import nn
from models import UNet


def get_model(model_name, num_classes=1):
    if model_name == "UNet":
        return UNet(num_classes)


if __name__ == "__main__":
    model_name = "UNet"
    classes = 1
    model = get_model(model_name, classes)
    size = 224
    image = torch.Tensor(
        3, 3, size, size
    )
    output = model(image)
    print(output.shape)
    pdb.set_trace()


""" footnotes
"""
