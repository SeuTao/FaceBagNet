import cv2
from .data_helper import *
from utils import *

class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index=-1, image_size=128, augment = None, augmentor = None, balance = True):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

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

        elif self.mode == 'val':
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            print('set dataset mode: test')

        elif self.mode == 'train':
            self.train_list = load_train_list()

            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)

            if self.balance:
                self.train_list = transform_balance(self.train_list)
            print('set dataset mode: train')

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
                color, depth, ir, label = tmp_list[pos]
            else:
                color,depth,ir,label = self.train_list[index]

        elif self.mode == 'val':
            color,depth,ir,label = self.val_list[index]

        elif self.mode == 'test':
            color, depth, ir = self.test_list[index]
            test_id = color + ' ' + depth + ' ' + ir

        if self.modality=='color':
            img_path = os.path.join(DATA_ROOT, color)
        elif self.modality=='depth':
            img_path = os.path.join(DATA_ROOT, depth)
        elif self.modality=='ir':
            img_path = os.path.join(DATA_ROOT, ir)

        image = cv2.imread(img_path,1)
        image = cv2.resize(image,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3))

            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
            n = len(image)
            image = np.concatenate(image,axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'test':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
            n = len(image)
            image = np.concatenate(image,axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), test_id


    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    from augmentation import color_augumentor
    augment = color_augumentor
    dataset = FDDataset(mode = 'train', fold_index=-1, image_size=32,  augment=augment)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label.shape)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


