import cv2
import glob
import numpy as np

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