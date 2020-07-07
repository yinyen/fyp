from preprocessing.load import load_img
import os
import glob
import cv2
main_dir = "../all_train/*/*/*.jpeg"
files = glob.glob(main_dir)
x = len(files)

# print(x)
# print(files[:10])
# print([j.replace("all_train", "resize_train") for j in files[:10]])

os.makedirs("../resize_train", exist_ok = True)
os.makedirs("../resize_train/train", exist_ok = True)
os.makedirs("../resize_train/val", exist_ok = True)
os.makedirs("../resize_train/test", exist_ok = True)
for i in range(5):
    os.makedirs(f"../resize_train/train/{i}", exist_ok = True)
    os.makedirs(f"../resize_train/val/{i}", exist_ok = True)
    os.makedirs(f"../resize_train/test/{i}", exist_ok = True)

size = 250
for f in files:
    f2 = f.replace("all_train", "resize_train")
    r = load_img(f, IMG_HEIGHT=size, IMG_WIDTH=size, rescale = False)
    print(r)
    cv2.imwrite(f2, r)
    # raise Exception("stop")
print("done")
#35126