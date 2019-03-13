import os
from utils import *
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
BatchNorm2d = nn.BatchNorm2d
from model.FaceBagNet_model_A import Net
from model.backbone.FaceBagNet import SEModule

###########################################################################################3
class FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2):
        super(FusionNet,self).__init__()

        self.color_moudle  = Net(num_class=num_class,is_first_bn=True)
        self.depth_moudle = Net(num_class=num_class,is_first_bn=True)
        self.ir_moudle = Net(num_class=num_class,is_first_bn=True)

        self.color_SE = SEModule(512,reduction=16)
        self.depth_SE = SEModule(512,reduction=16)
        self.ir_SE = SEModule(512,reduction=16)

        self.bottleneck = nn.Sequential(nn.Conv2d(512*3, 128*3, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(128*3),
                                         nn.ReLU(inplace=True))

        self.res_0 = self._make_layer(BasicBlock, 128*3, 256, 2, stride=2)
        self.res_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        color = x[:, 0:3,:,:]
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]

        color_feas = self.color_moudle.forward_res3(color)
        depth_feas = self.depth_moudle.forward_res3(depth)
        ir_feas = self.ir_moudle.forward_res3(ir)

        color_feas = self.color_SE(color_feas)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        fea = self.bottleneck(fea)

        x = self.res_0(fea)
        x = self.res_1(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x,None,None

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    num_class = 2
    net = Net(num_class)
    print(net)

########################################################################################
if __name__ == '__main__':
    import os
    run_check_net()
    print( 'sucessful!')