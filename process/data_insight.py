import os
import random
from utils import *

TRN_IMGS_DIR = '/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Training/'
TST_IMGS_DIR = '/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Val/'

LIST_DIR = r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/image_list'

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

def create_5fold_IDS():
    ID_list = os.listdir(os.path.join(TRN_IMGS_DIR, 'real_part'))
    random.shuffle(ID_list)

    for i in range(5):
        val_part = ID_list[i*60:(i+1)*60]

        val_part_dict = {}
        for tmp in val_part:
            val_part_dict[tmp] = 1

        print(val_part_dict)
        print(len(val_part_dict))
        load_train_val_list(val_part_dict, i)

    print('done')

def load_fold_list(fold_index=0, all = False):

    train_fold_list = load(os.path.join(LIST_DIR,'train_fold'+str(fold_index)+'.txt'))
    val_fold_list = load(os.path.join(LIST_DIR,'val_fold'+str(fold_index)+'.txt'))

    train_fold_list = add_id_label(train_fold_list)
    val_fold_list = add_id_label(val_fold_list)

    if all:
        train_fold_list =  train_fold_list + val_fold_list

    return train_fold_list, val_fold_list

def load_test_list(label_dir = '/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/Val_label/20190110_val_real'):

    label_dict = load_test_label(dir = label_dir)
    list = []

    f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/val_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')

        color = line[0]
        color_name = os.path.split(color)[1]

        if color_name in label_dict:
            line.append('1')
            list.append(line)
        else:
            line.append('0')
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

def submission(probs, outname):
    f = open('/data1/shentao/DATA/CVPR19_FaceAntiSpoofing/val_public_list.txt')
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

def add_id_label(list):
    id_dict = load(os.path.join(LIST_DIR, 'ID_dict.txt'))
    # print(id_dict)

    new_list = []
    for tmp in list:
        name_tmp = tmp[0].split('/')[2]
        tmp.append(id_dict[name_tmp])
        new_list.append(tmp)

    # print(new_list)

    return new_list






if __name__ == '__main__':
    # create_5fold_IDS()
    # load_train_val_list()
    # load_fold_list(fold_index=0)
    # load_test_list()
    # submission(None,'tmp.txt')

    # train_IDs()
    # train_list ,_ = load_fold_list()
    # add_id_label(train_list)
    tmp,_ = load_fold_list()
    print(tmp[0])
#
    # id_list = load( os.path.join(LIST_DIR,'ID_list.txt'))
    # print(id_list)
