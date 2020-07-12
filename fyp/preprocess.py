from multiprocessing import Pool
import glob, os
from tqdm import tqdm
from preprocessing.image_preprocess import open_image, load_transform_image
size = 400
data_dir = "../all_train"
out_dir = "../all_train_400"

os.makedirs(out_dir, exist_ok=True)
for i in range(5):
    os.makedirs(f"{out_dir}/{i}", exist_ok=True)
x = glob.glob(f"{data_dir}/*/*.jpeg")


def pre_process(i):
    # old_filename = i
    new_filename = i.replace("all_train", "all_train_400")
    # j+=1
    # if j %1000 == 0:
    #     print("Done:", j, new_filename)
    if os.path.exists(new_filename):
        return 

    img = load_transform_image(i,size)
    if img is None:
        print("Error ====")
    else:
        img.save(new_filename)

if __name__ == '__main__':
    pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    pool.map(pre_process, x)