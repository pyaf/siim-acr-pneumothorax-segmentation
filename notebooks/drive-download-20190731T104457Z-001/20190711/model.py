from common import *
from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d

# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
# https://github.com/jfzhang95/pytorch-deeplab-xception

#BatchNorm2d = nn.BatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

IMAGE_HEIGHT, IMAGE_WIDTH = 1024, 1024
LOGIT_HEIGHT, LOGIT_WIDTH = IMAGE_HEIGHT//4, IMAGE_WIDTH//4


# -----
CONVERSION = [
    'encode0.0.weight',  	 'encode0.0.weight',
    'encode0.1.weight',  	 'encode0.1.weight',
    'encode0.1.bias',  	 'encode0.1.bias',
    'encode0.1.running_mean',  	 'encode0.1.running_mean',
    'encode0.1.running_var',  	 'encode0.1.running_var',
    'encode1.0.conv1.weight',  	 'encode1.0.conv1.weight',
    'encode1.0.bn1.weight',  	 'encode1.0.bn1.weight',
    'encode1.0.bn1.bias',  	 'encode1.0.bn1.bias',
    'encode1.0.bn1.running_mean',  	 'encode1.0.bn1.running_mean',
    'encode1.0.bn1.running_var',  	 'encode1.0.bn1.running_var',
    'encode1.0.conv2.weight',  	 'encode1.0.conv2.weight',
    'encode1.0.bn2.weight',  	 'encode1.0.bn2.weight',
    'encode1.0.bn2.bias',  	 'encode1.0.bn2.bias',
    'encode1.0.bn2.running_mean',  	 'encode1.0.bn2.running_mean',
    'encode1.0.bn2.running_var',  	 'encode1.0.bn2.running_var',
    'encode1.0.conv3.weight',  	 'encode1.0.conv3.weight',
    'encode1.0.bn3.weight',  	 'encode1.0.bn3.weight',
    'encode1.0.bn3.bias',  	 'encode1.0.bn3.bias',
    'encode1.0.bn3.running_mean',  	 'encode1.0.bn3.running_mean',
    'encode1.0.bn3.running_var',  	 'encode1.0.bn3.running_var',
    'encode1.0.shortcut.0.weight',  	 'encode1.0.shortcut.0.weight',
    'encode1.0.shortcut.1.weight',  	 'encode1.0.shortcut.1.weight',
    'encode1.0.shortcut.1.bias',  	 'encode1.0.shortcut.1.bias',
    'encode1.0.shortcut.1.running_mean',  	 'encode1.0.shortcut.1.running_mean',
    'encode1.0.shortcut.1.running_var',  	 'encode1.0.shortcut.1.running_var',
    'encode1.1.conv1.weight',  	 'encode1.1.conv1.weight',
    'encode1.1.bn1.weight',  	 'encode1.1.bn1.weight',
    'encode1.1.bn1.bias',  	 'encode1.1.bn1.bias',
    'encode1.1.bn1.running_mean',  	 'encode1.1.bn1.running_mean',
    'encode1.1.bn1.running_var',  	 'encode1.1.bn1.running_var',
    'encode1.1.conv2.weight',  	 'encode1.1.conv2.weight',
    'encode1.1.bn2.weight',  	 'encode1.1.bn2.weight',
    'encode1.1.bn2.bias',  	 'encode1.1.bn2.bias',
    'encode1.1.bn2.running_mean',  	 'encode1.1.bn2.running_mean',
    'encode1.1.bn2.running_var',  	 'encode1.1.bn2.running_var',
    'encode1.1.conv3.weight',  	 'encode1.1.conv3.weight',
    'encode1.1.bn3.weight',  	 'encode1.1.bn3.weight',
    'encode1.1.bn3.bias',  	 'encode1.1.bn3.bias',
    'encode1.1.bn3.running_mean',  	 'encode1.1.bn3.running_mean',
    'encode1.1.bn3.running_var',  	 'encode1.1.bn3.running_var',
    'encode1.2.conv1.weight',  	 'encode1.2.conv1.weight',
    'encode1.2.bn1.weight',  	 'encode1.2.bn1.weight',
    'encode1.2.bn1.bias',  	 'encode1.2.bn1.bias',
    'encode1.2.bn1.running_mean',  	 'encode1.2.bn1.running_mean',
    'encode1.2.bn1.running_var',  	 'encode1.2.bn1.running_var',
    'encode1.2.conv2.weight',  	 'encode1.2.conv2.weight',
    'encode1.2.bn2.weight',  	 'encode1.2.bn2.weight',
    'encode1.2.bn2.bias',  	 'encode1.2.bn2.bias',
    'encode1.2.bn2.running_mean',  	 'encode1.2.bn2.running_mean',
    'encode1.2.bn2.running_var',  	 'encode1.2.bn2.running_var',
    'encode1.2.conv3.weight',  	 'encode1.2.conv3.weight',
    'encode1.2.bn3.weight',  	 'encode1.2.bn3.weight',
    'encode1.2.bn3.bias',  	 'encode1.2.bn3.bias',
    'encode1.2.bn3.running_mean',  	 'encode1.2.bn3.running_mean',
    'encode1.2.bn3.running_var',  	 'encode1.2.bn3.running_var',
    'encode2.0.conv1.weight',  	 'encode2.0.conv1.weight',
    'encode2.0.bn1.weight',  	 'encode2.0.bn1.weight',
    'encode2.0.bn1.bias',  	 'encode2.0.bn1.bias',
    'encode2.0.bn1.running_mean',  	 'encode2.0.bn1.running_mean',
    'encode2.0.bn1.running_var',  	 'encode2.0.bn1.running_var',
    'encode2.0.conv2.weight',  	 'encode2.0.conv2.weight',
    'encode2.0.bn2.weight',  	 'encode2.0.bn2.weight',
    'encode2.0.bn2.bias',  	 'encode2.0.bn2.bias',
    'encode2.0.bn2.running_mean',  	 'encode2.0.bn2.running_mean',
    'encode2.0.bn2.running_var',  	 'encode2.0.bn2.running_var',
    'encode2.0.conv3.weight',  	 'encode2.0.conv3.weight',
    'encode2.0.bn3.weight',  	 'encode2.0.bn3.weight',
    'encode2.0.bn3.bias',  	 'encode2.0.bn3.bias',
    'encode2.0.bn3.running_mean',  	 'encode2.0.bn3.running_mean',
    'encode2.0.bn3.running_var',  	 'encode2.0.bn3.running_var',
    'encode2.0.shortcut.0.weight',  	 'encode2.0.shortcut.0.weight',
    'encode2.0.shortcut.1.weight',  	 'encode2.0.shortcut.1.weight',
    'encode2.0.shortcut.1.bias',  	 'encode2.0.shortcut.1.bias',
    'encode2.0.shortcut.1.running_mean',  	 'encode2.0.shortcut.1.running_mean',
    'encode2.0.shortcut.1.running_var',  	 'encode2.0.shortcut.1.running_var',
    'encode2.1.conv1.weight',  	 'encode2.1.conv1.weight',
    'encode2.1.bn1.weight',  	 'encode2.1.bn1.weight',
    'encode2.1.bn1.bias',  	 'encode2.1.bn1.bias',
    'encode2.1.bn1.running_mean',  	 'encode2.1.bn1.running_mean',
    'encode2.1.bn1.running_var',  	 'encode2.1.bn1.running_var',
    'encode2.1.conv2.weight',  	 'encode2.1.conv2.weight',
    'encode2.1.bn2.weight',  	 'encode2.1.bn2.weight',
    'encode2.1.bn2.bias',  	 'encode2.1.bn2.bias',
    'encode2.1.bn2.running_mean',  	 'encode2.1.bn2.running_mean',
    'encode2.1.bn2.running_var',  	 'encode2.1.bn2.running_var',
    'encode2.1.conv3.weight',  	 'encode2.1.conv3.weight',
    'encode2.1.bn3.weight',  	 'encode2.1.bn3.weight',
    'encode2.1.bn3.bias',  	 'encode2.1.bn3.bias',
    'encode2.1.bn3.running_mean',  	 'encode2.1.bn3.running_mean',
    'encode2.1.bn3.running_var',  	 'encode2.1.bn3.running_var',
    'encode2.2.conv1.weight',  	 'encode2.2.conv1.weight',
    'encode2.2.bn1.weight',  	 'encode2.2.bn1.weight',
    'encode2.2.bn1.bias',  	 'encode2.2.bn1.bias',
    'encode2.2.bn1.running_mean',  	 'encode2.2.bn1.running_mean',
    'encode2.2.bn1.running_var',  	 'encode2.2.bn1.running_var',
    'encode2.2.conv2.weight',  	 'encode2.2.conv2.weight',
    'encode2.2.bn2.weight',  	 'encode2.2.bn2.weight',
    'encode2.2.bn2.bias',  	 'encode2.2.bn2.bias',
    'encode2.2.bn2.running_mean',  	 'encode2.2.bn2.running_mean',
    'encode2.2.bn2.running_var',  	 'encode2.2.bn2.running_var',
    'encode2.2.conv3.weight',  	 'encode2.2.conv3.weight',
    'encode2.2.bn3.weight',  	 'encode2.2.bn3.weight',
    'encode2.2.bn3.bias',  	 'encode2.2.bn3.bias',
    'encode2.2.bn3.running_mean',  	 'encode2.2.bn3.running_mean',
    'encode2.2.bn3.running_var',  	 'encode2.2.bn3.running_var',
    'encode3.0.conv1.weight',  	 'encode3.0.conv1.weight',
    'encode3.0.bn1.weight',  	 'encode3.0.bn1.weight',
    'encode3.0.bn1.bias',  	 'encode3.0.bn1.bias',
    'encode3.0.bn1.running_mean',  	 'encode3.0.bn1.running_mean',
    'encode3.0.bn1.running_var',  	 'encode3.0.bn1.running_var',
    'encode3.0.conv2.weight',  	 'encode3.0.conv2.weight',
    'encode3.0.bn2.weight',  	 'encode3.0.bn2.weight',
    'encode3.0.bn2.bias',  	 'encode3.0.bn2.bias',
    'encode3.0.bn2.running_mean',  	 'encode3.0.bn2.running_mean',
    'encode3.0.bn2.running_var',  	 'encode3.0.bn2.running_var',
    'encode3.0.conv3.weight',  	 'encode3.0.conv3.weight',
    'encode3.0.bn3.weight',  	 'encode3.0.bn3.weight',
    'encode3.0.bn3.bias',  	 'encode3.0.bn3.bias',
    'encode3.0.bn3.running_mean',  	 'encode3.0.bn3.running_mean',
    'encode3.0.bn3.running_var',  	 'encode3.0.bn3.running_var',
    'encode3.0.shortcut.0.weight',  	 'encode3.0.shortcut.0.weight',
    'encode3.0.shortcut.1.weight',  	 'encode3.0.shortcut.1.weight',
    'encode3.0.shortcut.1.bias',  	 'encode3.0.shortcut.1.bias',
    'encode3.0.shortcut.1.running_mean',  	 'encode3.0.shortcut.1.running_mean',
    'encode3.0.shortcut.1.running_var',  	 'encode3.0.shortcut.1.running_var',
    'encode3.1.conv1.weight',  	 'encode3.1.conv1.weight',
    'encode3.1.bn1.weight',  	 'encode3.1.bn1.weight',
    'encode3.1.bn1.bias',  	 'encode3.1.bn1.bias',
    'encode3.1.bn1.running_mean',  	 'encode3.1.bn1.running_mean',
    'encode3.1.bn1.running_var',  	 'encode3.1.bn1.running_var',
    'encode3.1.conv2.weight',  	 'encode3.1.conv2.weight',
    'encode3.1.bn2.weight',  	 'encode3.1.bn2.weight',
    'encode3.1.bn2.bias',  	 'encode3.1.bn2.bias',
    'encode3.1.bn2.running_mean',  	 'encode3.1.bn2.running_mean',
    'encode3.1.bn2.running_var',  	 'encode3.1.bn2.running_var',
    'encode3.1.conv3.weight',  	 'encode3.1.conv3.weight',
    'encode3.1.bn3.weight',  	 'encode3.1.bn3.weight',
    'encode3.1.bn3.bias',  	 'encode3.1.bn3.bias',
    'encode3.1.bn3.running_mean',  	 'encode3.1.bn3.running_mean',
    'encode3.1.bn3.running_var',  	 'encode3.1.bn3.running_var',
    'encode3.2.conv1.weight',  	 'encode3.2.conv1.weight',
    'encode3.2.bn1.weight',  	 'encode3.2.bn1.weight',
    'encode3.2.bn1.bias',  	 'encode3.2.bn1.bias',
    'encode3.2.bn1.running_mean',  	 'encode3.2.bn1.running_mean',
    'encode3.2.bn1.running_var',  	 'encode3.2.bn1.running_var',
    'encode3.2.conv2.weight',  	 'encode3.2.conv2.weight',
    'encode3.2.bn2.weight',  	 'encode3.2.bn2.weight',
    'encode3.2.bn2.bias',  	 'encode3.2.bn2.bias',
    'encode3.2.bn2.running_mean',  	 'encode3.2.bn2.running_mean',
    'encode3.2.bn2.running_var',  	 'encode3.2.bn2.running_var',
    'encode3.2.conv3.weight',  	 'encode3.2.conv3.weight',
    'encode3.2.bn3.weight',  	 'encode3.2.bn3.weight',
    'encode3.2.bn3.bias',  	 'encode3.2.bn3.bias',
    'encode3.2.bn3.running_mean',  	 'encode3.2.bn3.running_mean',
    'encode3.2.bn3.running_var',  	 'encode3.2.bn3.running_var',
    'encode4.0.conv1.weight',  	 'encode4.0.conv1.weight',
    'encode4.0.bn1.weight',  	 'encode4.0.bn1.weight',
    'encode4.0.bn1.bias',  	 'encode4.0.bn1.bias',
    'encode4.0.bn1.running_mean',  	 'encode4.0.bn1.running_mean',
    'encode4.0.bn1.running_var',  	 'encode4.0.bn1.running_var',
    'encode4.0.conv2.weight',  	 'encode4.0.conv2.weight',
    'encode4.0.bn2.weight',  	 'encode4.0.bn2.weight',
    'encode4.0.bn2.bias',  	 'encode4.0.bn2.bias',
    'encode4.0.bn2.running_mean',  	 'encode4.0.bn2.running_mean',
    'encode4.0.bn2.running_var',  	 'encode4.0.bn2.running_var',
    'encode4.0.conv3.weight',  	 'encode4.0.conv3.weight',
    'encode4.0.bn3.weight',  	 'encode4.0.bn3.weight',
    'encode4.0.bn3.bias',  	 'encode4.0.bn3.bias',
    'encode4.0.bn3.running_mean',  	 'encode4.0.bn3.running_mean',
    'encode4.0.bn3.running_var',  	 'encode4.0.bn3.running_var',
    'encode4.1.conv1.weight',  	 'encode4.1.conv1.weight',
    'encode4.1.bn1.weight',  	 'encode4.1.bn1.weight',
    'encode4.1.bn1.bias',  	 'encode4.1.bn1.bias',
    'encode4.1.bn1.running_mean',  	 'encode4.1.bn1.running_mean',
    'encode4.1.bn1.running_var',  	 'encode4.1.bn1.running_var',
    'encode4.1.conv2.weight',  	 'encode4.1.conv2.weight',
    'encode4.1.bn2.weight',  	 'encode4.1.bn2.weight',
    'encode4.1.bn2.bias',  	 'encode4.1.bn2.bias',
    'encode4.1.bn2.running_mean',  	 'encode4.1.bn2.running_mean',
    'encode4.1.bn2.running_var',  	 'encode4.1.bn2.running_var',
    'encode4.1.conv3.weight',  	 'encode4.1.conv3.weight',
    'encode4.1.bn3.weight',  	 'encode4.1.bn3.weight',
    'encode4.1.bn3.bias',  	 'encode4.1.bn3.bias',
    'encode4.1.bn3.running_mean',  	 'encode4.1.bn3.running_mean',
    'encode4.1.bn3.running_var',  	 'encode4.1.bn3.running_var',
    'encode4.2.conv1.weight',  	 'encode4.2.conv1.weight',
    'encode4.2.bn1.weight',  	 'encode4.2.bn1.weight',
    'encode4.2.bn1.bias',  	 'encode4.2.bn1.bias',
    'encode4.2.bn1.running_mean',  	 'encode4.2.bn1.running_mean',
    'encode4.2.bn1.running_var',  	 'encode4.2.bn1.running_var',
    'encode4.2.conv2.weight',  	 'encode4.2.conv2.weight',
    'encode4.2.bn2.weight',  	 'encode4.2.bn2.weight',
    'encode4.2.bn2.bias',  	 'encode4.2.bn2.bias',
    'encode4.2.bn2.running_mean',  	 'encode4.2.bn2.running_mean',
    'encode4.2.bn2.running_var',  	 'encode4.2.bn2.running_var',
    'encode4.2.conv3.weight',  	 'encode4.2.conv3.weight',
    'encode4.2.bn3.weight',  	 'encode4.2.bn3.weight',
    'encode4.2.bn3.bias',  	 'encode4.2.bn3.bias',
    'encode4.2.bn3.running_mean',  	 'encode4.2.bn3.running_mean',
    'encode4.2.bn3.running_var',  	 'encode4.2.bn3.running_var',
    'aspp.atrous0.0.weight',  	 'aspp.atrous0.0.weight',
    'aspp.atrous0.1.weight',  	 'aspp.atrous0.1.weight',
    'aspp.atrous0.1.bias',  	 'aspp.atrous0.1.bias',
    'aspp.atrous0.1.running_mean',  	 'aspp.atrous0.1.running_mean',
    'aspp.atrous0.1.running_var',  	 'aspp.atrous0.1.running_var',
    'aspp.atrous1.module.0.weight',  	 'aspp.atrous1.module.0.weight',
    'aspp.atrous1.module.1.weight',  	 'aspp.atrous1.module.1.weight',
    'aspp.atrous1.module.1.bias',  	 'aspp.atrous1.module.1.bias',
    'aspp.atrous1.module.1.running_mean',  	 'aspp.atrous1.module.1.running_mean',
    'aspp.atrous1.module.1.running_var',  	 'aspp.atrous1.module.1.running_var',
    'aspp.atrous2.module.0.weight',  	 'aspp.atrous2.module.0.weight',
    'aspp.atrous2.module.1.weight',  	 'aspp.atrous2.module.1.weight',
    'aspp.atrous2.module.1.bias',  	 'aspp.atrous2.module.1.bias',
    'aspp.atrous2.module.1.running_mean',  	 'aspp.atrous2.module.1.running_mean',
    'aspp.atrous2.module.1.running_var',  	 'aspp.atrous2.module.1.running_var',
    'aspp.atrous3.module.0.weight',  	 'aspp.atrous3.module.0.weight',
    'aspp.atrous3.module.1.weight',  	 'aspp.atrous3.module.1.weight',
    'aspp.atrous3.module.1.bias',  	 'aspp.atrous3.module.1.bias',
    'aspp.atrous3.module.1.running_mean',  	 'aspp.atrous3.module.1.running_mean',
    'aspp.atrous3.module.1.running_var',  	 'aspp.atrous3.module.1.running_var',
    'aspp.atrous4.module.1.weight',  	 'aspp.atrous4.module.1.weight',
    'aspp.atrous4.module.2.weight',  	 'aspp.atrous4.module.2.weight',
    'aspp.atrous4.module.2.bias',  	 'aspp.atrous4.module.2.bias',
    'aspp.atrous4.module.2.running_mean',  	 'aspp.atrous4.module.2.running_mean',
    'aspp.atrous4.module.2.running_var',  	 'aspp.atrous4.module.2.running_var',
    'aspp.combine.0.weight',  	 'aspp.combine.0.weight',
    'aspp.combine.1.weight',  	 'aspp.combine.1.weight',
    'aspp.combine.1.bias',  	 'aspp.combine.1.bias',
    'aspp.combine.1.running_mean',  	 'aspp.combine.1.running_mean',
    'aspp.combine.1.running_var',  	 'aspp.combine.1.running_var',
    'decode.side.0.weight',  	 'decoder.side.0.weight',
    'decode.side.1.weight',  	 'decoder.side.1.weight',
    'decode.side.1.bias',  	 'decoder.side.1.bias',
    'decode.side.1.running_mean',  	 'decoder.side.1.running_mean',
    'decode.side.1.running_var',  	 'decoder.side.1.running_var',
    'decode.logit.0.weight',  	 'decoder.logit.0.weight',
    'decode.logit.1.weight',  	 'decoder.logit.1.weight',
    'decode.logit.1.bias',  	 'decoder.logit.1.bias',
    'decode.logit.1.running_mean',  	 'decoder.logit.1.running_mean',
    'decode.logit.1.running_var',  	 'decoder.logit.1.running_var',
    'decode.logit.4.weight',  	 'decoder.logit.4.weight',
    'decode.logit.5.weight',  	 'decoder.logit.5.weight',
    'decode.logit.5.bias',  	 'decoder.logit.5.bias',
    'decode.logit.5.running_mean',  	 'decoder.logit.5.running_mean',
    'decode.logit.5.running_var',  	 'decoder.logit.5.running_var',
    'logit.weight',  	 'decoder.logit.8.weight',
    'logit.bias',  	 'decoder.logit.8.bias',
]


def load_pretrain(net, pretrain_file, conversion=CONVERSION):

    #raise NotImplementedError
    print('\tload pretrain_file: %s' % pretrain_file)

    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1, 2)
    for key, pretrain_key in conversion:
        if any(s in key for s in []):
            continue

        # print('\t\t',key)
        print('\t\t', '%-48s  %-24s  <---  %-32s  %-24s' % (
            key, str(state_dict[key].shape),
            pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
        ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d' %
          len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d' % len(state_dict.keys()))
    print('loaded    = %d' % i)
    print('')


# -----

class ASPPConv(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ASPPConv, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation,
                      dilation=dilation, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.module(x)
        return x


class ASPPPool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPPPool, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.module(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel=256, rate=[6, 12, 8]):
        super(ASPP, self).__init__()

        self.atrous0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.atrous1 = ASPPConv(in_channel, out_channel, rate[0])
        self.atrous2 = ASPPConv(in_channel, out_channel, rate[1])
        self.atrous3 = ASPPConv(in_channel, out_channel, rate[2])
        self.atrous4 = ASPPPool(in_channel, out_channel)

        self.combine = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):

        x = torch.cat([
            self.atrous0(x),
            self.atrous1(x),
            self.atrous2(x),
            self.atrous3(x),
            self.atrous4(x),
        ], 1)
        x = self.combine(x)
        return x

# https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/


class Decode(nn.Module):
    def __init__(self, in_channel, side_channel, channel, num_class=1):
        super(Decode, self).__init__()

        self.side = nn.Sequential(
            nn.Conv2d(side_channel, 48, kernel_size=1, bias=False),
            BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.logit = nn.Sequential(
            nn.Conv2d(in_channel+48, channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(channel, channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x, side):
        side = self.side(side)

        x = F.interpolate(
            x, size=side.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, side), 1)
        logit = self.logit(x)

        return logit

#########################################################################################
# resent50


class Bottleneck(nn.Module):

    def __init__(self, in_channel, channel, out_channel, stride=1, dilation=1, is_shortcut=False):
        super(Bottleneck, self).__init__()
        self.is_shortcut = is_shortcut
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channel)

        if self.is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                BatchNorm2d(out_channel),
            )

    def forward(self, x):

        if self.is_shortcut:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)

        x = self.bn2(self.conv2(x))
        x = F.relu(x, inplace=True)

        x = self.bn3(self.conv3(x)) + shortcut
        x = F.relu(x, inplace=True)

        return x


def make_encode_layer(in_channel, channel, out_channel, stride=1, dilation=1, num=1):

    is_shortcut = stride != 1 or in_channel != out_channel

    layer = [
        Bottleneck(in_channel,  channel, out_channel, stride,
                   dilation[0], is_shortcut=is_shortcut)
    ]
    for i in range(1, num):
        layer.append(
            Bottleneck(out_channel, channel, out_channel, stride=1,
                       dilation=dilation[i], is_shortcut=False)
        )

    return nn.Sequential(*layer)


#########################################################################################

class DeepLab3Plus(nn.Module):
    def __init__(self, in_channel=1, num_class=1):
        super(DeepLab3Plus, self).__init__()

        # dilated resnet50 backbone [3, 4, 6, 3] ----
        self.encode0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.encode1 = make_encode_layer(
            64,  64, 128, stride=1, dilation=[1, 1, 1], num=3)
        self.encode2 = make_encode_layer(
            128,  64, 256, stride=2, dilation=[1, 1, 1, 1], num=3)
        self.encode3 = make_encode_layer(
            256, 128, 512, stride=1, dilation=[2, 2, 2, 2, ], num=3)
        self.encode4 = make_encode_layer(
            512, 256, 512, stride=1, dilation=[4, 8, 16], num=3)

        self.aspp = ASPP(512, 128, rate=[6, 12, 8])
        self.decode = Decode(128, 128, 128, num_class)

        self.logit = nn.Conv2d(128, num_class, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.encode0(x)  # ;print('encode0  :', x.shape)
        x = self.encode1(x)
        side = x  # ;print('encode1  :', x.shape)
        x = self.encode2(x)  # ;print('encode2  :', x.shape)
        x = self.encode3(x)  # ;print('encode3  :', x.shape)
        x = self.encode4(x)  # ;print('encode4  :', x.shape)

        x = self.aspp(x)  # ;print('aspp    :', x.shape)
        x = self.decode(x, side)  # ;print('decoder :', x.shape)
        logit = self.logit(x)
        #x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        return logit


Net = DeepLab3Plus

'''
encode0  : torch.Size([1, 64, 256, 256])
encode1  : torch.Size([1, 256, 256, 256])
encode2  : torch.Size([1, 512, 128, 128])
encode3  : torch.Size([1, 1024, 128, 128])
encode4  : torch.Size([1, 2048, 128, 128])

aspp     : torch.Size([2, 256, 128, 128])
decoder  : torch.Size([2, 1, 256, 256])

#===========================================
input:  torch.Size([2, 1, 1024, 1024])
logit:  torch.Size([2, 1, 256, 256])

'''


def criterion(logit, truth, reduction='mean'):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_weight = pos.sum().item() + 1e-12
    neg_weight = neg.sum().item() + 1e-12
    loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

    # loss=loss.mean()
    return loss


def metric(logit, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)

    with torch.no_grad():
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(logit.shape == truth.shape)

        probability = torch.sigmoid(logit)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        #print(len(neg_index), len(pos_index))

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


# check #################################################################
def run_check_net():

    batch_size = 2
    C, H, W = 1, 1024, 1024
    num_class = 1

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = torch.from_numpy(input).float().cuda()

    net = Net(in_channel=C, num_class=num_class).cuda()
    with torch.no_grad():
        logit = net(input)

    # print(net)
    print('input: ', input.shape)
    print('logit: ', logit.shape)

    if 0:
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        sorted(keys)
        for k in keys:
            if '.num_batches_tracked' in k:
                continue

            weight = ''
            if k.endswith('.weight'):
                weight = str(state_dict[k].shape)
            print(' \'%s\',  \t%s' % (k, weight))
    if 0:
        state_dict = torch.load(
            '/root/share/project/kaggle/2019/chest/result/deeplab3plus-00-256x256/checkpoint/00292500_model.old.pth', map_location=lambda storage, loc: storage
        )
        keys = list(state_dict.keys())
        sorted(keys)
        for k in keys:
            if '.num_batches_tracked' in k:
                continue

            weight = ''
            if k.endswith('.weight'):
                weight = str(state_dict[k].shape)
            print(' \'%s\',  \t%s' % (k, weight))
    if 1:
        pretrain_file = \
            '/root/share/project/kaggle/2019/chest/result/deeplab3plus-00-256x256/checkpoint/00292500_model.old.pth'
        load_pretrain(net, pretrain_file, conversion=CONVERSION)

        torch.save(net.state_dict(),
                   '/root/share/project/kaggle/2019/chest/result/deeplab3plus-00-256x256/checkpoint/00292500_model.pth')


def run_check_train():

    batch_size = 2
    C, H, W = 1, 1024, 1024
    num_class = 1

    num_component = np.array([0, 1])
    #assert (np.any(num_component==0))

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    truth = np.random.uniform(0, 1, (batch_size, C, H//4, W//4))
    truth[num_component == 0, ...] = 0

    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()

    net = Net(in_channel=C, num_class=num_class).cuda()
    net = net.eval()

    logit = net(input)
    loss = criterion(logit, truth)
    dice, dice_neg, dice_pos, num_neg, num_pos = metric(logit, truth)
    print('loss = %0.5f' % loss.item())
    print('dice, dice_neg, dice_pos = %0.5f, %0.5f, %0.5f' %
          (dice, dice_neg, dice_pos))
    print('num_neg, num_pos = %d, %d' % (num_neg, num_pos))
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)

    print('-------------------------------------------------')
    print('[iter ]  loss        dice    dice_neg dice_pos   ')
    print('-------------------------------------------------')
    # [00000]  0.70383  | 0.26128, 0.00000, 0.46449

    i = 0
    optimizer.zero_grad()
    while i <= 500:
        net.train()
        optimizer.zero_grad()

        logit = net(input)
        loss = criterion(logit, truth)
        dice, dice_neg, dice_pos, num_neg, num_pos = metric(logit, truth)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[%05d] %8.5f  | %0.5f, %0.5f, %0.5f ' % (
                i,
                loss.item(),
                dice, dice_neg, dice_pos
            ))
        i = i+1
    print('')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()
    # run_check_train()
