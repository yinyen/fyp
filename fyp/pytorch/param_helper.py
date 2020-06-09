import os
import yaml


def import_config(file):
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def dump_config(output_path, doc):
    with open(output_path, 'w') as f:
        _ = yaml.dump(doc, f)


def create_dir(train_name, main_dir = "torch_models"):
    path = f'{main_dir}/{train_name}'
    initial_train_name = train_name
    j = 0 
    while os.path.exists(path):
        train_name = initial_train_name + "_v" + str(j).zfill(2)
        path = f'{main_dir}/{train_name}'
        j = j + 1
    os.makedirs(path)
    return train_name

def create_main_dir(train_name):
    path = train_name
    initial_train_name = train_name
    j = 0 
    while os.path.exists(path):
        train_name = initial_train_name + "_v" + str(j).zfill(2)
        path = train_name
        j = j + 1
    os.makedirs(path)
    return train_name


