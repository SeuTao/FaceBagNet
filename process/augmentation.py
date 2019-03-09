from imgaug import augmenters as iaa
import random
import math
import numpy as np
import torch
import numpy as np
from PIL import Image
import cv2
import os
# percent_crop = 0.6
# print('!!!!!!!!!! RC:' + str(percent_crop))

RESIZE_SIZE = 112

def random_cropping(image, target_shape=(32, 32, 3), is_random = True):
    image = cv2.resize(image,(RESIZE_SIZE,RESIZE_SIZE))
    target_h, target_w,_ = target_shape
    height, width, _ = image.shape

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    return zeros

def TTA_0_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    # print(start_x)
    # print(start_y)
    # starts = [[start_x, start_y],
    #           [0, 0],
    #           [ 2 * start_x, 0],
    #           [0, 2 * start_y],
    #           [2 * start_x, 2 * start_y]]

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              # [start_x + 32, start_y + 32],
              # [start_x - 32, start_y - 32],
              # [start_x - 32, start_y + 32],
              # [start_x + 32, start_y - 32],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()
        zeros = np.fliplr(zeros)
        image_flip = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    # images = []
    # image_ = image.copy()
    # zeros = image_[start_x:start_x + target_w, start_x: start_x+target_h, :]
    # images.append(zeros.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    # zeros = np.fliplr(zeros.copy())
    # images.append(zeros.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    return images

def TTA_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()
        zeros = np.fliplr(zeros)
        image_flip = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images

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

def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio)
    new_w = int(w*ratio)

    img = cv2.resize(img, (new_w,new_h))
    img = cv2.resize(img, (w, h))
    return img

def color_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

def depth_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

def ir_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])
        image =  augment_img.augment_image(image)
        image = TTA_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

# def depth_augumentor(image, is_infer=False, augment = None):
#     if is_infer:
#         if augment is None:
#             flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
#                 0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0
#         else:
#             flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
#                 augment[0], augment[1],augment[2],augment[3],augment[4]
#
#         augment_img = iaa.Sequential([
#             iaa.Fliplr(flip_prob),
#             iaa.Crop(percent=(top_percent, right_percent, bottom_percent, left_percent), keep_size=True)
#         ])
#
#         return augment_img.augment_image(image)
#     else:
#         up_rand = np.random.random()
#         right_rand = np.random.random()
#
#         augment_img = iaa.Sequential([
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5),
#             iaa.Affine(rotate=(-30, 30)),
#             iaa.Crop(percent=(up_rand * percent_crop, right_rand * percent_crop,
#                               (1 - up_rand) * percent_crop, (1 - right_rand) * percent_crop), keep_size=True),
#
#         ], random_order=True)
#
#         image = augment_img.augment_image(image)
#         image = random_resize(image)
#         return image
#
# def ir_augumentor(image, is_infer=False, augment = None):
#
#     if is_infer:
#         if augment is None:
#             flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
#                 0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0, percent_crop/2.0
#         else:
#             flip_prob, top_percent, right_percent, bottom_percent, left_percent = \
#                 augment[0], augment[1],augment[2],augment[3],augment[4]
#
#         augment_img = iaa.Sequential([
#             iaa.Fliplr(flip_prob),
#             iaa.Crop(percent=(top_percent, right_percent, bottom_percent, left_percent), keep_size=True)
#         ])
#
#         return augment_img.augment_image(image)
#
#     else:
#         up_rand = np.random.random()
#         right_rand = np.random.random()
#
#         augment_img = iaa.Sequential([
#             iaa.Fliplr(0.5),
#             iaa.Flipud(0.5),
#             iaa.Affine(rotate=(-30, 30)),
#             iaa.Crop(percent=(up_rand * percent_crop, right_rand * percent_crop,
#                               (1 - up_rand) * percent_crop, (1 - right_rand) * percent_crop), keep_size=True),
#
#         ], random_order=True)
#
#         image = augment_img.augment_image(image)
#         image = random_resize(image)
#         # image = random_keep(image)
#         return image

def render_imgs():
    dir = r'./tmp'
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = r'/data1/shentao/Projects/CVPR19FaceAntiSpoofing/0104_out/fp/000133-color.jpg'
    img = cv2.imread(path,1)
    img = cv2.resize(img,(112,112))

    for i in range(20):
        # img_ = random_erasing(img.copy(),probability=1.0,r1=0.5)
        # img_ = random_keep(img.copy(),probability=1.0)

        img_ = random_cropping(img, target_shape=(48, 48, 3), is_random = False)
        # img_ = color_augumentor(img.copy())
        cv2.imwrite(os.path.join(dir,str(i)+'.jpg'),img_)

    return

if __name__ == '__main__':
    render_imgs()
