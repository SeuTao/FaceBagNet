##  https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
#
# Lovasz-Softmax and Jaccard hinge loss in PyTorch
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)

from common import *



# Computes gradient of the Lovasz extension w.r.t sorted errors
# See Alg. 1 in paper

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
    truth = truth.float()
    sign  = 2. * truth - 1.
    hinge = (1. - logit * sign)
    hinge, permutation = torch.sort(hinge, dim=0, descending=True)
    hinge = F.relu(hinge)

    truth = truth[permutation.data]
    gradient = compute_lovasz_gradient(truth)

    loss = torch.dot(hinge, gradient)
    return loss


def lovasz_loss(logit, truth, mode='hinge', is_average=True):

    if mode=='hinge':
        lovasz_one = lovasz_hinge_one
    elif mode=='soft_hinge':
        lovasz_one = lovasz_soft_hinge_one
    elif mode=='logistic':
        lovasz_one = lovasz_logistic_one
    elif mode=='exp':
        lovasz_one = lovasz_exp_one
    else:
        raise NotImplementedError

    batch_size,C,H,W = truth.shape
    loss = torch.zeros(batch_size).cuda()
    for b in range(batch_size):
        l, t = logit[b].view(-1), truth[b].view(-1)
        loss[b] = lovasz_one(l, t)

    if is_average:
        loss = loss.sum()/batch_size

    return loss


#####################################################################


#https://www.groundai.com/project/a-new-smooth-approximation-to-the-zero-one-loss-with-a-probabilistic-interpretation/
def lovasz_logistic_one(logit , truth):
    lamda = 8
    logit = torch.clamp(logit,-4,4)

    truth = truth.float()
    sign  = 2. * truth - 1.
    logistic = torch.log(1+torch.exp(lamda*(1- logit*sign)))/lamda
    logistic, permutation = torch.sort(logistic, dim=0, descending=True)

    truth = truth[permutation.data]
    gradient = compute_lovasz_gradient(truth)

    loss = torch.dot(logistic, gradient)
    return loss




def lovasz_exp_one(logit , truth):

    truth = truth.float()
    sign  = 2. * truth - 1.
    exp   = torch.exp(-logit*sign)
    exp, permutation = torch.sort(exp, dim=0, descending=True)

    truth = truth[permutation.data]
    gradient = compute_lovasz_gradient(truth)

    loss = torch.dot(exp, gradient)
    return loss



# https://github.com/pytorch/pytorch/blob/master/torch/legacy/nn/SoftPlus.py
def lovasz_soft_hinge_one(logit , truth):

    truth = truth.float()
    sign  = 2. * truth - 1.
    hinge = (1. - logit * sign)
    hinge, permutation = torch.sort(hinge, dim=0, descending=True)
    hinge = nn.Softplus()(hinge)

    truth = truth[permutation.data]
    gradient = compute_lovasz_gradient(truth)

    loss = torch.dot(hinge, gradient)
    return loss


