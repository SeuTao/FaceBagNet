from imgaug import augmenters as iaa
import random
import math
import numpy as np
import torch
import numpy as np
from PIL import Image
import cv2
import os

# class RandomErasing(object):
#     '''
#     Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
#     -------------------------------------------------------------------------------------
#     probability: The probability that the operation will be performed.
#     sl: min erasing area
#     sh: max erasing area
#     r1: min aspect ratio
#     mean: erasing value
#     -------------------------------------------------------------------------------------

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.5, r1 = 0.5, channel = 3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)

            noise = np.random.random((h,w,channel))*255
            noise = noise.astype(np.uint8)

            if img.shape[2] == channel:
                img[x1:x1 + h, y1:y1 + w, :] = noise
            else:
                print('wrong')
                return
            return img

    return img

def random_keep(img, probability = 0.5, min_ratio = 0.4, max_ratio = 0.4, channel = 3, mode = 'black'):
    if random.uniform(0, 1) > probability:
        return img

    ratio = 0.4

    origin_w = img.shape[0]
    origin_h = img.shape[1]

    h = int(round(origin_h*ratio))
    w = int(round(origin_w*ratio))

    if w < img.shape[1] and h < img.shape[0]:
        x1 = random.randint(0, img.shape[0] - h)
        y1 = random.randint(0, img.shape[1] - w)

        if mode == 'black':
            noise = np.zeros(img.shape)
        elif mode == 'random':
            noise = np.random.random((img.shape[0],img.shape[1],channel))*255

        noise = noise.astype(np.uint8)
        noise[x1:x1 + h, y1:y1 + w, :] = img[x1:x1 + h, y1:y1 + w, :]
        return noise



def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio)
    new_w = int(w*ratio)

    # print(ratio)
    # print(new_h)
    # print(new_w)

    img = cv2.resize(img, (new_w,new_h))
    img = cv2.resize(img, (w, h))

    return img

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

percent_crop = 0.2

def color_augumentor(image, is_infer=False, augment = None):
    if is_infer:
        if augment is None:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0
        else:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                augment[0], augment[1],augment[2],augment[3],augment[4]

        augment_img = iaa.Sequential([
            iaa.Fliplr(flip_prob),
            iaa.Crop(percent=(top_percent, right_percent, bottom_percent, left_percent), keep_size=True)
        ])

        return augment_img.augment_image(image)

    else:
        up_rand = np.random.random()
        right_rand = np.random.random()

        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.Crop(percent=(up_rand * percent_crop, right_rand * percent_crop,
                              (1 - up_rand) * percent_crop, (1 - right_rand) * percent_crop), keep_size=True),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_keep(image)
        return image

def depth_augumentor(image, is_infer=False, augment = None):

    if is_infer:
        if augment is None:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0
        else:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                augment[0], augment[1],augment[2],augment[3],augment[4]

        augment_img = iaa.Sequential([
            iaa.Fliplr(flip_prob),
            iaa.Crop(percent=(top_percent, right_percent, bottom_percent, left_percent), keep_size=True)
        ])

        return augment_img.augment_image(image)
    else:
        up_rand = np.random.random()
        right_rand = np.random.random()

        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.Crop(percent=(up_rand * percent_crop, right_rand * percent_crop,
                              (1 - up_rand) * percent_crop, (1 - right_rand) * percent_crop), keep_size=True),

        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_keep(image)
        return image

def ir_augumentor(image, is_infer=False, augment = None):

    if is_infer:
        if augment is None:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0
        else:
            flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
                augment[0], augment[1],augment[2],augment[3],augment[4]

        augment_img = iaa.Sequential([
            iaa.Fliplr(flip_prob),
            iaa.Crop(percent=(top_percent, right_percent, bottom_percent, left_percent), keep_size=True)
        ])

        return augment_img.augment_image(image)

    else:
        up_rand = np.random.random()
        right_rand = np.random.random()

        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.Crop(percent=(up_rand * percent_crop, right_rand * percent_crop,
                              (1 - up_rand) * percent_crop, (1 - right_rand) * percent_crop), keep_size=True),

        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_keep(image)
        return image

def render_imgs():
    dir = r'./tmp'
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/0104_out/fp/000133-color.jpg'
    img = cv2.imread(path,1)
    img = cv2.resize(img,(112,112))

    for i in range(20):
        # img_ = random_erasing(img.copy(),probability=1.0,r1=0.5)

        img_ = random_keep(img.copy(),probability=1.0)
        # img_ = color_augumentor(img.copy())
        cv2.imwrite(os.path.join(dir,str(i)+'.jpg'),img_)

    return

if __name__ == '__main__':
    render_imgs()
