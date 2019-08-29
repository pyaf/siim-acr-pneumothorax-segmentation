import torch
from torch import nn
import torch.nn.functional as F


def WBCE(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def wbce(logit, truth):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if 0:
        loss = loss.mean()

    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12 # pos count
        neg_weight = neg.sum().item() + 1e-12 # neg count
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

    return loss

def wsd(logit_pixel, truth_pixel):
    batch_size = len(logit_pixel)
    logit = logit_pixel.view(batch_size,-1)
    truth = truth_pixel.view(batch_size,-1)
    assert(logit.shape==truth.shape)

    loss = soft_dice_criterion(logit, truth)

    loss = loss.mean()
    return loss

def soft_dice_criterion(logit, truth, weight=[0.2,0.8]):

    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    w = truth.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss



def wlova(logit, truth, margin=[1,5]):

    def compute_lovasz_gradient(truth): #sorted
        truth_sum    = truth.sum()
        intersection = truth_sum - truth.cumsum(0)
        union        = truth_sum + (1 - truth).cumsum(0)
        jaccard      = 1. - intersection / union
        T = len(truth)
        jaccard[1:T] = jaccard[1:T] - jaccard[0:T-1]

        gradient = jaccard
        return gradient

    def lovasz_hinge_one(logit , truth):

        m = truth.detach()
        m = m*(margin[1]-margin[0])+margin[0]

        truth = truth.float()
        sign  = 2. * truth - 1.
        hinge = (m - logit * sign)
        hinge, permutation = torch.sort(hinge, dim=0, descending=True)
        hinge = F.relu(hinge)

        truth = truth[permutation.data]
        gradient = compute_lovasz_gradient(truth)

        loss = torch.dot(hinge, gradient)
        return loss

    #----
    lovasz_one = lovasz_hinge_one

    batch_size = len(truth)
    loss = torch.zeros(batch_size).cuda()
    for b in range(batch_size):
        l, t = logit[b].view(-1), truth[b].view(-1)
        loss[b] = lovasz_one(l, t)


    return loss.mean()

def criterion_pixel(logit_pixel, truth_pixel):
    batch_size = len(logit_pixel)
    logit = logit_pixel.view(batch_size,-1)
    truth = truth_pixel.view(batch_size,-1)
    assert(logit.shape==truth.shape)

    loss = lovasz_loss(logit, truth)

    loss = loss.mean()
    return loss





class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        N = targets.size(0)
        preds = torch.sigmoid(logits)
        EPSILON = 1

        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = (preds_flat * targets_flat).sum()#.float()
        union = (preds_flat + targets_flat).sum()#.float()

        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = 1 - loss / N
        return loss


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

