import torch.nn.functional as F
from torch.autograd import Variable


def criterion_GAN(pred, target_is_real, use_sigmoid=True):
    """use sigmoid or ls-gan"""
    if use_sigmoid:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(0.))

        loss = F.binary_cross_entropy(pred, target_var)
    else:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).fill_(0.))

        loss = F.mse_loss(pred, target_var)

    return loss


def discriminate(net, crit, fake, real):
    pred_fake = net(fake)
    loss_fake = crit(pred_fake, False)

    pred_true = net(real)
    loss_true = crit(pred_true, True)

    return loss_fake, loss_true, pred_fake, pred_true
