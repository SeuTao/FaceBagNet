# from common import *
from torch.autograd import Variable
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()


def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)

    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def bce_criterion(logit, truth, is_average=True):
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduce=is_average)
    return loss

from sklearn.metrics import f1_score

def do_valid( net, valid_loader, criterion ):
    valid_num  = 0
    losses   = []

    f1 = []
    target_list = []
    output_list = []
    for input, truth in valid_loader:
        input = input.cuda()
        truth = truth.cuda()

        input = to_var(input)
        truth = to_var(truth)

        logit,_   = net(input)
        loss    = criterion(logit, truth, False)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())

        target_list.append(truth.cpu().data.numpy())
        output_list.append(logit.sigmoid().cpu().data.numpy())


    assert(valid_num == len(valid_loader.sampler))

    target = np.concatenate(target_list)
    output = np.concatenate(output_list)

    f1_max = 0.0
    thres_max = 0.0
    for thres in np.arange(0.0, 0.8, 0.05):
        f1 = f1_score(target, output > thres, average='macro')
        if f1_max < f1:
            f1_max = f1
            thres_max = thres

    loss    = np.concatenate(losses)
    loss    = loss.mean()

    valid_loss = np.array([ loss, f1_max, thres_max])
    return valid_loss


def do_valid_8TTA(net, valid_loader, criterion):
    valid_num = 0
    losses = []

    f1 = []
    target_list = []
    output_list = []
    for input, truth in valid_loader:
        img = input.cpu().data.numpy()
        img1 = np.array(img)
        img2 = np.array(img1)[:, :, ::-1, :]
        img3 = np.array(img1)[:, :, :, ::-1]
        img4 = np.array(img2)[:, :, :, ::-1]

        img_flip = np.concatenate([img1, img2, img3, img4])
        img_flip_trans = np.transpose(img_flip, (0, 1, 3, 2))

        img_all = np.concatenate([img_flip, img_flip_trans], axis=0)
        y_pred, _ = net(to_var(torch.FloatTensor(img_all)))

        batch_size = len(input)
        # y_pred = y_pred.sigmoid().cpu().data.numpy()

        logit = y_pred[:(0 + 1) * batch_size] + \
                y_pred[(1) * batch_size:(1 + 1) * batch_size] + \
                y_pred[(2) * batch_size:(2 + 1) * batch_size] + \
                y_pred[(3) * batch_size:(3 + 1) * batch_size] + \
                y_pred[(4) * batch_size:(4 + 1) * batch_size] + \
                y_pred[(5) * batch_size:(5 + 1) * batch_size] + \
                y_pred[(6) * batch_size:(6 + 1) * batch_size] + \
                y_pred[(7) * batch_size:]

        logit /= 8.0


        # input = input.cuda()
        truth = truth.cuda()
        #
        # input = to_var(input)
        truth = to_var(truth)
        #
        # logit, _ = net(input)

        loss = criterion(logit, truth, False)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())

        target_list.append(truth.cpu().data.numpy())
        output_list.append(logit.sigmoid().cpu().data.numpy())

    assert (valid_num == len(valid_loader.sampler))

    target = np.concatenate(target_list)
    output = np.concatenate(output_list)

    f1_max = 0.0
    thres_max = 0.0
    for thres in np.arange(0.0, 0.8, 0.05):
        f1 = f1_score(target, output > thres, average='macro')
        if f1_max < f1:
            f1_max = f1
            thres_max = thres

    loss = np.concatenate(losses)
    loss = loss.mean()

    valid_loss = np.array([loss, f1_max, thres_max])
    return valid_loss


def load_CLASS_NAME():

    label_list_path = r'/data2/backup/Projects/Kaggle_Whale/image_list/label_list.txt'
    f = open(label_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    id_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        id = line[0]
        index = int(line[1])
        label_dict[index] = id
        id_dict[id] = index
    return label_dict, id_dict

def load_train_map():

    train_image_list_path = r'/data2/backup/Projects/Kaggle_Whale/image_list/train_image_list.txt'
    f = open(train_image_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        index = int(line[1])
        id = line[2]

        label_dict[img_name] = [index,id]


    return label_dict

def prob_to_csv_top5(prob, key_id, name):
    CLASS_NAME = load_CLASS_NAME()

    prob = np.asarray(prob)
    print(prob.shape)

    top = np.argsort(-prob,1)[:,:5]
    word = []
    index = 0

    rs = []

    for (t0,t1,t2,t3,t4) in top:
        word.append(
            CLASS_NAME[t0] + ' ' + \
            CLASS_NAME[t1] + ' ' + \
            CLASS_NAME[t2])

        top_k_label_name = r''

        label = CLASS_NAME[t0]
        score = prob[index][t0]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t1]
        score = prob[index][t1]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t2]
        score = prob[index][t2]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t3]
        score = prob[index][t3]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t4]
        score = prob[index][t4]
        top_k_label_name += label + ' ' + str(score) + ' '

        # print(top_k_label_name)
        rs.append(top_k_label_name)
        index += 1
        # break

    pd.DataFrame({'key_id':key_id, 'word':rs}).to_csv( '{}.csv'.format(name), index=None)


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    dict = load_train_map()
    print(dict)