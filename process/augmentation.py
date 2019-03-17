from imgaug import augmenters as iaa
import math
import cv2
from data_helper import *

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


def TTA_5_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],]

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
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images

def TTA_18_cropps(image, target_shape=(32, 32, 3)):
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

def TTA_36_cropps(image, target_shape=(32, 32, 3)):
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
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

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
        image = TTA_36_cropps(image, target_shape)
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
        image = TTA_36_cropps(image, target_shape)
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
        image = TTA_36_cropps(image, target_shape)
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

