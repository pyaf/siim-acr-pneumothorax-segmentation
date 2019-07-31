# https://github.com/junfu1115/DANet

from common import *
from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d

#BatchNorm2d = nn.BatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


LOGIT_TO_IMAGE_SCALE = 4


##############################################################

class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(Conv2dBn, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              padding=padding, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Basic(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, is_shortcut=False):
        super(Basic, self).__init__()
        self.is_shortcut = in_channel != out_channel or stride != 1
        if self.is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          padding=0, stride=stride, bias=False),
                BatchNorm2d(out_channel)
            )

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channel)

    def forward(self, x):

        if self.is_shortcut:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.bn2(self.conv2(x)) + shortcut
        x = F.relu(x, inplace=True)
        return x


class Decode(nn.Module):
    def __init__(self, in_channel, channel, out_channel):
        super(Decode, self).__init__()
        self.top = nn.Sequential(
            nn.Conv2d(in_channel+channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, e):
        batch_size, C, H, W = e.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([x, e], 1)
        x = self.top(x)
        return x


##############################################################


class UNet(nn.Module):

    def __init__(self, in_channel=1, num_class=1):
        super(UNet, self).__init__()

        self.encode = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2,)
            )
        ])
        for in_channel, out_channel, stride, num_block in [
            [64,          64,     1,       3],
            [64,         128,     2,       4],
            [128,         256,     2,       6],
            [256,         512,     2,       3],
            [512,         512,     2,       2],
            [512,         512,     2,       2],
        ]:
            self.encode.append(
                nn.Sequential(
                    Basic(in_channel, out_channel,  stride=stride, ),
                    *[Basic(out_channel, out_channel,  stride=1,) for i in range(1, num_block)]
                )
            )

        self.decode = nn.ModuleList([
            Decode(512, 512, 256),
            Decode(256, 512, 256),
            Decode(256, 256, 128),
            Decode(128, 128, 64),
            Decode(64, 64, 32),
        ])

        self.logit = nn.Conv2d(
            32, num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.encode[0](x)  # print('encode[0] :', x.shape)
        x = self.encode[1](x)
        e1 = x  # ; print('encode[1] :', x.shape)
        x = self.encode[2](x)
        e2 = x  # ; print('encode[2] :', x.shape)
        x = self.encode[3](x)
        e3 = x  # ; print('encode[3] :', x.shape)
        x = self.encode[4](x)
        e4 = x  # ; print('encode[4] :', x.shape)
        x = self.encode[5](x)
        e5 = x  # ; print('encode[5] :', x.shape)
        x = self.encode[6](x)  # print('encode[6] :', x.shape)

        x = self.decode[0](x, e5)  # ; print('decode[0] :', x.shape)
        x = self.decode[1](x, e4)  # ; print('decode[1] :', x.shape)
        x = self.decode[2](x, e3)  # ; print('decode[2] :', x.shape)
        x = self.decode[3](x, e2)  # ; print('decode[3] :', x.shape)
        x = self.decode[4](x, e1)  # ; print('decode[4] :', x.shape)

        #x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit


Net = UNet


'''

encode[0] : torch.Size([1, 64, 256, 256])
encode[1] : torch.Size([1, 64, 256, 256])
encode[2] : torch.Size([1, 128, 128, 128])
encode[3] : torch.Size([1, 256, 64, 64])
encode[4] : torch.Size([1, 512, 32, 32])
encode[5] : torch.Size([1, 512, 16, 16])
encode[6] : torch.Size([1, 512, 8, 8])
decode[0] : torch.Size([1, 256, 16, 16])
decode[1] : torch.Size([1, 256, 32, 32])
decode[2] : torch.Size([1, 128, 64, 64])
decode[3] : torch.Size([1, 64, 128, 128])
decode[4] : torch.Size([1, 32, 256, 256])

position_attention: torch.Size([1, 256, 128, 128])
channel_attention : torch.Size([1, 256, 128, 128])

input:  torch.Size([1, 1, 1024, 1024])
logit:  torch.Size([1, 1, 256, 256])


'''


def criterion(logit, truth):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if 0:
        loss = loss.mean()

    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

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


###################################################################################


def run_check_net():

    batch_size = 1
    C, H, W = 1, 1024, 1024
    num_class = 1

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = torch.from_numpy(input).float().cuda()

    net = Net(in_channel=C, num_class=num_class).cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('input: ', input.shape)
    print('logit: ', logit.shape)
    # print(net)


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

    # exit(0)

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

    # run_check_net()
    run_check_train()

    print('\nsucess!')
