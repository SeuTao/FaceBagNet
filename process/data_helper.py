import os
import random
from utils import *

TRN_IMGS_DIR = '/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Training/'
TST_IMGS_DIR = '/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Val/'
LIST_DIR = r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/image_list'
DATA_ROOT = r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing'

RESIZE_SIZE = 112

def load_train_val_list(ID_dict, fold_index):
    train_fold_list = []
    val_fold_list = []

    f = open(r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/train_list.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line_ = line.strip()
        line_ = line_.split(' ')
        color_name = line_[0]
        ID = color_name.split('/')[2]

        if ID in ID_dict:
            val_fold_list.append(line.strip().split(' '))
        else:
            train_fold_list.append(line.strip().split(' '))

    print(len(lines))
    print(len(train_fold_list))

    save(train_fold_list,os.path.join(LIST_DIR,'train_fold'+str(fold_index)+'.txt'))
    save(val_fold_list,os.path.join(LIST_DIR,'val_fold'+str(fold_index)+'.txt'))

def load_fold_list(fold_index=0, all = False):
    train_fold_list = load(os.path.join(LIST_DIR,'train_fold'+str(fold_index)+'.txt'))
    val_fold_list = load(os.path.join(LIST_DIR,'val_fold'+str(fold_index)+'.txt'))

    if all:
        train_fold_list =  train_fold_list + val_fold_list

    return train_fold_list, val_fold_list

def load_val_list():
    list = []
    f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list():
    list = []
    f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_label(dir):
    print(dir)
    pos_list = os.listdir(dir)
    pos_dict = {}
    for tmp in pos_list:
        pos_dict[tmp] = 1

    print('load from: ' + dir +' test_pos_num:'+str(len(pos_dict)))
    return pos_dict

def transform_balance(train_list):
    print('balance!!!!!!!!')

    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/val_public_list.txt')
    else:
        f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list

def train_IDs():
    id_list = os.listdir(TRN_IMGS_DIR+'fake_part')
    print(len(id_list))

    id_list = [[id_list[i],i] for i in range(len(id_list))]
    print(id_list)

    id_dict={}
    for name, id in id_list:
        id_dict[name] = id
        print(name)
        print(id)

    save(id_dict, os.path.join(LIST_DIR,'ID_dict.txt'))

if __name__ == '__main__':
    load_test_list()

