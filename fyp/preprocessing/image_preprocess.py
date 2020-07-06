import cv2
import glob
import numpy as np
from PIL import Image, ImageOps
from skimage import io, transform


def open_image(img_name):
    image2 = Image.open(img_name)
    return image2


def transform_img(img, func):
    '''transform image wrapper with try catch'''
    try:
        img2 = func(img)
        x,y = img.size[0], img.size[1]
        x2,y2 = img2.size[0], img2.size[1]
        if x2 < x*0.6 or y2 < y*0.6:
            print("Crop beyond limit!", func)
            return img
        else:
            return img2
    except:
        print("Error on func!", func)
        return img


def pad_to_square(image2):
    '''pad image to make image square'''
    try:
        new_size = image2.size
        max_size = max(image2.size)
        desired_size = max_size
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(image2, padding)
    except:
        new_im = image2
    return new_im


def remove_border_2(img):
    '''remove border using difference in border intensity'''
    x = np.array(img)
    QH = x.max(axis = 2).mean(axis = 0)
    QV = x.max(axis = 2).mean(axis = 1)
    DH = np.diff(QH)
    DV = np.diff(QV)
    LEFT = np.argmax(DH) if np.max(DH) > 1 else 0
    RIGHT = np.argmin(DH) if np.min(DH) < -1 else -1
    DOWN = np.argmax(DV) if np.max(DV) > 1 else 0
    UP = np.argmin(DV) if np.min(DV) < -1 else -1
    x2 = x[DOWN:UP, LEFT:RIGHT, :]
    im = Image.fromarray(x2)
    return im


def remove_border_1(img):
    '''remove low intensity border pixel'''
    x = np.array(img)
    x2 = x.mean(axis = 2)
    p = np.percentile(x2, q = 5)
    z = (x2 <= p).mean(axis = 0) > 0.99
    LEFT = z.argmin()
    RIGHT = -z[::-1].argmin() - 1 
    z = (x2 <= p).mean(axis = 1) > 0.99
    DOWN = z.argmin()
    UP = -z[::-1].argmin() - 1
    x = x[DOWN:UP, LEFT:RIGHT, :]
    im = Image.fromarray(x)
    return im


def resize(img, size):
    try:
        x = np.array(img)
        z = x[x.shape[0] // 2, :, :].sum(1)
        r = (z > z.mean() / 10).sum() / 2
        s = size * 1.0 / (2*r)
        x = cv2.resize(x, (0,0), fx = s , fy = s)
        im = Image.fromarray(x)
        return im
    except:
        return img


def load_transform_image(img_name, size):
    image = open_image(img_name)
    image = transform_img(image, remove_border_1)
    image = pad_to_square(image)
    image = resize(image, size)
    return image


####################################
###### ARCHIVE######################
####################################
def scaleRadius(img,scale):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0,0), fx = s , fy = s)

def preprocess_eye(f, scale = 300):
    try:
        # scale = 300
        a = cv2.imread(f)

        a = scaleRadius(a,scale)
        
        #substract local average colour
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128 )
        
        # remove 10% outer part of image
        b = np.zeros(a.shape)
        cv2.circle(b,(a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9),(1, 1, 1), -1, 8, 0)
        # a = a * b + 128 * (1 - b)
        a = a * b 
        return a.astype(int)
    except:
        print("Error at:", f)
        return None

# for f in glob.glob("train/*.jpeg") + glob.glob("test/*.jpeg"):
#     try:
#         a = cv2.imread(f)

#         #scale image to the given same radius
#         a = scaleRadius(a,scale)

#         #substract local average colour
#         a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128 )

#         # remove 10% outer part of image
#         b = np.zeros(a.shape)
#         cv2.circle(b,(a.shape[1] / 2, a.shape[0] / 2), int(scale * 0.9),(1, 1, 1), -1, 8, 0)
#         a = a * b + 128 * (1 - b)
#         cv2.imwrite(str(scale) + "_" + f, a)
#     except:
#         print f