import os
import glob
import shutil
files = glob.glob("../all_train/*/*.jpeg")

K = 0
COUNTER = 0

os.makedirs(f"../train/", exist_ok=True)
os.makedirs(f"../train/{K}", exist_ok=True)
os.makedirs(f"../train/{K}/{K}/", exist_ok=True)
for j in files:
    f = os.path.basename(j)
    shutil.copy2(j, f"../train/{K}/{K}/{f}")
    COUNTER+=1
    if COUNTER > 3000:
        COUNTER = 0
        K += 1
        os.makedirs(f"../train/{K}", exist_ok=True)
        os.makedirs(f"../train/{K}/{K}/", exist_ok=True)
