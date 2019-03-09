from utils import *
import torchvision.models as tvm
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from backbone.senet import se_resnext18_32x4d

from modified_resnet import Modified_resnet18

BatchNorm2d = nn.BatchNorm2d

from  torchvision.models.resnet import BasicBlock,ResNet
def resnet_tiny( **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

###########################################################################################3
class Net(nn.Module):
    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict['module.'+key]

        self.load_state_dict(state_dict)
        print('load: '+pretrain_file)


    def __init__(self, num_class=2, id_class = 300, is_first_bn = False):
        super(Net,self).__init__()

        self.is_first_bn = is_first_bn

        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3)

        self.encoder  = se_resnext18_32x4d()

        self.conv1 = self.encoder.layer0
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.fc = nn.Sequential(nn.Linear(2048, num_class))
        self.id_fc = nn.Sequential(nn.Linear(2048, id_class))

    def forward(self, x):
        batch_size,C,H,W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean=[0.485, 0.456, 0.406] #rgb
            std =[0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:,[0]]-mean[0])/std[0],
                (x[:,[1]]-mean[1])/std[1],
                (x[:,[2]]-mean[2])/std[2],
            ],1)

        x = self.conv1(x) #; print('e1',x.size())
        x = self.conv2(x) #; print('e2',x.size())
        x = self.conv3(x) #; print('e3',x.size())
        x = self.conv4(x) #; print('e4',x.size())
        x = self.conv5(x) #; print('e5',x.size())

        fea = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
        fea = F.dropout(fea, p=0.50, training=self.training)
        logit = self.fc(fea)
        logit_id = self.id_fc(fea)

        return logit, logit_id, fea

    def forward_res3(self, x):
        batch_size,C,H,W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean=[0.485, 0.456, 0.406] #rgb
            std =[0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:,[0]]-mean[0])/std[0],
                (x[:,[1]]-mean[1])/std[1],
                (x[:,[2]]-mean[2])/std[2],
            ],1)


        x = self.conv1(x) #; print('e1',x.size())
        x = self.conv2(x) #; print('e2',x.size())
        x = self.conv3(x) #; print('e3',x.size())

        return x

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
    batch_size = 32
    C,H,W = 3, 128, 128
    num_class = 340

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()

    input = to_var(input)
    truth = to_var(truth)

    #---
    criterion = softmax_cross_entropy_criterion
    net = Net(num_class).cuda()
    net.set_mode('backup')
    print(net)
    ## exit(0)

########################################################################################
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # '3,2,1,0'
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_net()
    print( 'sucessful!')