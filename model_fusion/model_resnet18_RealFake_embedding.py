from utils import *
import torchvision.models as tvm
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from model.modified_resnet import Modified_resnet18

BatchNorm2d = nn.BatchNorm2d


from  torchvision.models.resnet import BasicBlock,ResNet
def resnet_tiny( **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

###########################################################################################3
class Net(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        # keys = list(pretrain_state_dict.keys())
        # for key in keys:
        #     print(key)

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

        self.encoder  = tvm.resnet18(pretrained=False)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.real_embedding = nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(True))

        self.fake_embedding = nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(True))

        self.id_real_fc = nn.Sequential(nn.Linear(512, id_class))
        self.id_fake_fc = nn.Sequential(nn.Linear(512, id_class))

        # self.final_fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(512*2, num_class))
        self.final_fc_real = nn.Sequential(nn.Linear(512, 1))
        self.final_fc_fake = nn.Sequential(nn.Linear(512, 1))

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

        real_embedding = self.real_embedding(fea)
        fake_embedding = self.fake_embedding(fea)

        logit_id_real = self.id_real_fc(real_embedding)
        logit_id_fake = self.id_fake_fc(fake_embedding)

        real_1 = self.final_fc_real(real_embedding)
        fake_1 = self.final_fc_fake(fake_embedding)
        logit = torch.cat([real_1,fake_1],dim=1)

        # fc_embedding = torch.cat([real_embedding,fake_embedding], 1)
        # logit = self.final_fc(fc_embedding)

        return logit, logit_id_real, logit_id_fake

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