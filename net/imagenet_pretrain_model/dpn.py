from resnet32.net.imagenet_pretrain_model.common import *


##############################################################3
#  https://github.com/Cadene/pretrained-models.pytorch
#  https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/dpn.py

# 'dpn68b': {
#       'imagenet+5k': {
#           'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-84854c156.pth',
#           'input_space': 'RGB',
#           'input_size': [3, 224, 224],
#           'input_range': [0, 1],
#           'mean': [124 / 255, 117 / 255, 104 / 255],
#           'std': [1 / (.0167 * 255)] * 3,
#           'num_classes': 1000
#       }
#   },
# 'dpn92': {
#
#     'imagenet+5k': {
#         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth',
#         'input_space': 'RGB',
#         'input_size': [3, 224, 224],
#         'input_range': [0, 1],
#         'mean': [124 / 255, 117 / 255, 104 / 255],
#         'std': [1 / (.0167 * 255)] * 3,
#         'num_classes': 1000
#     }
# },


BatchNorm2d = SynchronizedBatchNorm2d
#BatchNorm2d = nn.BatchNorm2d


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4


        # conv1
        blocks1 = collections.OrderedDict()
        if small:
            blocks1['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks1['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        blocks2 = collections.OrderedDict()
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks2['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks2['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        blocks3 = collections.OrderedDict()
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks3['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks3['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        blocks4 = collections.OrderedDict()
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks4['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks4['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        blocks5 = collections.OrderedDict()
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks5['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks5['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        #blocks5['conv5_bn_ac'] = CatBnAct(in_chs)


        self.layer1 = nn.Sequential(blocks1)
        self.layer2 = nn.Sequential(blocks2)
        self.layer3 = nn.Sequential(blocks3)
        self.layer4 = nn.Sequential(blocks4)
        self.layer5 = nn.Sequential(blocks5)


        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        #self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x    = self.layer1(x)       ; print(x.shape)
        r, d = self.layer2(x)       ; print(r.shape,d.shape)
        r, d = self.layer3((r, d )) ; print(r.shape,d.shape)
        r, d = self.layer4((r, d )) ; print(r.shape,d.shape)
        r, d = self.layer5((r, d )) ; print(r.shape,d.shape)


        '''
        torch.Size([2, 10, 56, 56])
        torch.Size([2, 64, 56, 56]) torch.Size([2, 80, 56, 56])
        torch.Size([2, 128, 28, 28]) torch.Size([2, 192, 28, 28])
        torch.Size([2, 256, 14, 14]) torch.Size([2, 448, 14, 14])
        torch.Size([2, 512, 7, 7]) torch.Size([2, 320, 7, 7])
        '''

        #exit(0)
        # if not self.training and self.test_time_pool:
        #     x   = F.avg_pool2d(x, kernel_size=7, stride=1)
        #     out = self.classifier(x)
        #     # The extra test time pool should be pooling an img_size//32 - 6 size patch
        #     out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        # else:
        #     x = adaptive_avgmax_pool2d(x, pool_type='avg')
        #     out = self.classifier(x)
        # return out.view(out.size(0), -1)
        return r, d

#
# def dpn68b(num_classes=1000, pretrained=False, test_time_pool=True):
#     model = DPN(
#         small=True, num_init_features=10, k_r=128, groups=32,
#         b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
#         num_classes=num_classes, test_time_pool=test_time_pool)
#
#     if pretrained:
#         if model_urls['dpn68b-extra']:
#             model.load_state_dict(model_zoo.load_url(model_urls['dpn68b-extra']))
#         elif has_mxnet and os.path.exists('./pretrained/'):
#             convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68-extra')
#         else:
#             assert False, "Unable to load a pretrained model"
#     return model
#
# def dpn92(num_classes=1000, pretrained='imagenet+5k'):
#     model = DPN(
#         num_init_features=64, k_r=96, groups=32,
#         k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
#         num_classes=num_classes, test_time_pool=True)
#
#     if pretrained:
#         settings = pretrained_settings['dpn92'][pretrained]
#         assert num_classes == settings['num_classes'], \
#             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
#
#         model.load_state_dict(model_zoo.load_url(settings['url']))
#         model.input_space = settings['input_space']
#         model.input_size = settings['input_size']
#         model.input_range = settings['input_range']
#         model.mean = settings['mean']
#         model.std = settings['std']
#     return model



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    print( 'sucessful!')