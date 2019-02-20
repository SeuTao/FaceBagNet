
from utils import *
import random
import os
import cv2
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
from data_insight import *

TRAIN_DF  = []
TEST_DF   = []

DATA_ROOT = r'/data1/shentao/DATA/CVPR19_FaceAntiSpoofing'

def get_label_dict(csv):
    def get_label_dict(train_df):
        sample_dict = {}
        for i in range(28):
            sample_dict[i] = []

        label_dict = {}
        for name, lbl in zip(train_df['Id'], train_df['Target'].str.split(' ')):
            y = np.zeros(28)
            for key in lbl:
                y[int(key)] = 1
                sample_dict[int(key)].append(name)
            label_dict[name] = y
        return label_dict, sample_dict

    train_df = pd.read_csv(csv)
    label_dict, sample_dict = get_label_dict(train_df)
    return label_dict, sample_dict

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

    def __init__(self, mode, modality='color', fold_index='<NIL>', image_size=128, augment = None, augmentor = None, balance = True):
        super(FDDataset, self).__init__()

        print('fold: '+str(fold_index))
        print(modality)
        # print()
        self.mode       = mode
        self.modality = modality

        self.augment = augment
        self.augmentor = augmentor
        self.balance = balance

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

                if self.balance:
                    self.train_list = transform_balance(self.train_list)

                print('set dataset mode: train')

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
                color,depth,ir,label, id_label = self.train_list[index]

        elif self.mode == 'val':
            color,depth,ir,label, id_label = self.val_list[index]
        elif self.mode == 'test':
            color, depth, ir,label = self.test_list[index]
            test_id = color + ' ' + depth + ' ' + ir

        if self.modality=='color':
            img_path = os.path.join(DATA_ROOT, color)
        elif self.modality=='depth':
            img_path = os.path.join(DATA_ROOT, depth)
        elif self.modality=='ir':
            img_path = os.path.join(DATA_ROOT, ir)

        image = cv2.imread(img_path,1)
        image = cv2.resize(image,(112,112))

        if self.mode == 'train':
            image = self.augment(image)
        else:
            image = self.augment(image, is_infer = True, augment = self.augmentor)

        image = cv2.resize(image,(self.image_size,self.image_size))

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.reshape([self.channels, self.image_size, self.image_size])
        image = image / 255.0

        label = int(label)

        if self.mode == 'test':
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1])), test_id
        else:
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1])),torch.LongTensor(np.asarray(id_label).reshape([-1]))

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
        print(label.shape)
        break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


