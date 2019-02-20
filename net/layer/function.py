import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np
from scipy.sparse import coo_matrix


def coo_to_sparse_tensor(coo):
    index  = np.vstack([coo.row, coo.col])
    index  = torch.from_numpy(index).long()
    data   = torch.from_numpy(coo.data).float()
    tensor = torch.sparse.FloatTensor( index , data, torch.Size(coo.shape) )
    return tensor

class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()
    def forward(self, x):
        return x


##https://pytorch.org/docs/0.3.1/_modules/torch/nn/modules/padding.html
class ReflectiveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ReflectiveConv2d, self).__init__()

        self.pad  = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=0, stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


## group norm : https://github.com/kuangliu/pytorch-groupnorm/blob/master/groupnorm.py
#  "Group Normalization" - Yuxin Wu, Kaiming He, arxiv 2018
#  https://arxiv.org/abs/1803.08494

#  https://github.com/switchablenorms/Switchable-Normalization
#  "Differentiable Learning-to-Normalize via Switchable Normalization"
#    - Ping Luo and Jiamin Ren and Zhanglin Peng, arxiv 2018

class LayerNorm2d(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones (1,num_features,1,1))
        self.bias   = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1,1))
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight  = softmax(self.var_weight)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


###########################################################################################
# https://discuss.pytorch.org/t/how-to-flip-a-tensor-left-right/17062/2
# https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382/5

def torch_flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

###########################################################################################
## https://github.com/xbresson/spectral_graph_convnets

class sparse_mm(torch.autograd.Function):

    def forward(self, x_sparse, y):
        self.save_for_backward(x_sparse, y)
        z = torch.mm(x_sparse, y)
        return z

    def backward(self, grad_output):
        x_sparse, y = self.saved_tensors
        grad = grad_output.clone()
        dL_dx = torch.mm(grad, y.t())
        dL_dy = torch.mm(x_sparse.t(), grad )
        return dL_dx, dL_dy


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    N = 3
    D = 5
    F = 7

    idx = torch.LongTensor([[0, 1, 1, 2],
                            [2, 0, 2, 4]])
    value = torch.randn(4)
    x_sparse = torch.sparse.FloatTensor(idx, value, torch.Size([N, D]))
    x = x_sparse.to_dense()

    #x = torch.randn(N, D, requires_grad=True)
    w = torch.randn(F, D)
    y_true = torch.randn(N, F)

    x.requires_grad        =True
    x_sparse.requires_grad =True
    w.requires_grad        =True


    print('===  dense ===============================\n')

    print('before ------\n')
    print('x.grad\n',x.grad,'\n')
    print('w.grad\n',w.grad,'\n')


    y = torch.mm(x,w.t())
    loss = 0.5*((y-y_true)**2).sum()
    loss.backward(retain_graph=True )


    print('after ------\n')
    print('y\n',y,'\n')
    print('x.grad\n',x.grad,'\n')
    print('y.grad\n',y.grad,'\n')
    print('w.grad\n',w.grad,'\n')



    print('===  sparse ===============================\n')
    x.grad.zero_()
    w.grad.zero_()


    print('before ------\n')
    print('x_sparse.grad\n',x_sparse.grad,'\n')
    print('w.grad\n',w.grad,'\n')


    #y_sparse = torch.mm(x_sparse,w.t()) ##error
    y_sparse = sparse_mm()(x_sparse, w.t())
    loss_sparse = 0.5*((y_sparse-y_true)**2).sum()
    loss_sparse.backward(retain_graph=True)

    print('after ------\n')
    print('y_sparse\n',y_sparse,'\n')
    print('x_sparse.grad\n',x_sparse.grad,'\n')
    print('w.grad\n',w.grad,'\n')


