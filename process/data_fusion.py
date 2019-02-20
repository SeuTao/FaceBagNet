# from common import *
from utils import *
import random
import os
import cv2
from augmentation import *

TRAIN_DF  = []
TEST_DF   = []

from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

from data_insight import *

DATA_ROOT = r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing'

def get_sample_dict(label_dict, external_label_dict, image_list):
    sample_dict = {}
    for i in range(28):
        sample_dict[i] = []

    for name in image_list:
        if name in label_dict:
            y = label_dict[name]
        else:
            y = external_label_dict[name]

        for i in range(28):
            if y[i] == 1:
                sample_dict[i].append(name)

    c = 0
    for item in sample_dict:
        c += len(sample_dict[item])

    return sample_dict, c

class FDDataset(Dataset):

    def __init__(self, mode, modality='color', fold_index='<NIL>', image_size=128, augment = None, balance=False):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        self.augment = augment
        self.balance = balance

        # print()
        self.mode       = mode
        self.modality = modality

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index

        self.set_mode(self.mode,self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print('fold index set: ', fold_index)

        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            print('set dataset mode: test')
        else:
            if self.fold_index == -1:
                self.train_list, self.val_list = load_fold_list(0,all=True)
            else:
                self.train_list, self.val_list = load_fold_list(self.fold_index)

            if self.mode == 'train':
                random.shuffle(self.train_list)
                self.num_data = len(self.train_list)
                print('set dataset mode: train')

                if self.balance:
                    self.train_list = transform_balance(self.train_list)

            if self.mode == 'val':
                self.num_data = len(self.val_list)
                print('set dataset mode: val')

        print(self.num_data)


    def __getitem__(self, index):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':
            if self.balance:
                if random.randint(0,1)==0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0,len(tmp_list)-1)
                color, depth, ir, label, id_label = tmp_list[pos]
            else:
                color, depth, ir, label, id_label = self.train_list[index]

        elif self.mode == 'val':
            color,depth,ir,label, id_label = self.val_list[index]
        elif self.mode == 'test':
            color,depth,ir,label = self.test_list[index]
            test_id = color+' '+depth+' '+ir

        color = cv2.imread(os.path.join(DATA_ROOT, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT, depth),1)
        ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)

        # color = cv2.resize(color,(self.image_size,self.image_size))
        # depth = cv2.resize(depth,(self.image_size,self.image_size))
        # ir = cv2.resize(ir,(self.image_size,self.image_size))

        color = cv2.resize(color,(112,112))
        depth = cv2.resize(depth,(112,112))
        ir = cv2.resize(ir,(112,112))


        if self.mode == 'train':
            color = color_augumentor(color)
            depth = depth_augumentor(depth)
            ir = ir_augumentor(ir)

        elif self.mode == 'test':
            color = color_augumentor(color, is_infer=True,augment=self.augment)
            depth = depth_augumentor(depth, is_infer=True,augment=self.augment)
            ir = ir_augumentor(ir, is_infer=True,augment=self.augment)

        color = cv2.resize(color,(self.image_size,self.image_size))
        depth = cv2.resize(depth,(self.image_size,self.image_size))
        ir = cv2.resize(ir,(self.image_size,self.image_size))

        image = np.concatenate([color.reshape([self.image_size,self.image_size,3]),
                                depth.reshape([self.image_size,self.image_size,3]),
                                ir.reshape([self.image_size,self.image_size,3])],
                                axis=2)

        if self.mode == 'train':
            if random.randint(0,1) == 0:
                random_pos = random.randint(0, 2)
                if random.randint(0, 1) == 0:
                    image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(3):
                        if i!= random_pos:
                            image[:, :, 3 * i:3 * (i + 1)] = 0

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.reshape([self.channels*3, self.image_size, self.image_size])
        image = image / 255.0

        label = int(label)

        if self.mode == 'test':
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1])), test_id
        else:
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    dataset = FDDataset(mode = 'train', fold_index=0)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label)
        break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


