import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random


def initialize_dataset(main_data_dir = "./data3", batch_size = 64, size = 100, workers = 16, fix_sample = None, fix_sample_val = -1, force_random_sample = -1, use_train_dir = "full_train"):

    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(root=f'{main_data_dir}/{use_train_dir}', transform=transform_train)
    test_set = datasets.ImageFolder(root=f'{main_data_dir}/val', transform=transform_test)
    unseen_test_set = datasets.ImageFolder(root=f'{main_data_dir}/test', transform=transform_test)
    
    # calculate weights
    if fix_sample is not None:
        targets = np.array(train_set.targets)
        if force_random_sample != -1:
            idxs = [j for j in range(len(targets))]
            random.seed(123)
            random.shuffle(idxs)
            idxs = idxs[:force_random_sample]
            s_targets = targets[idxs]
        else:
            s_targets = targets

        samples_weight = np.array([1/np.mean(s_targets == i) for i in np.unique(s_targets)])

        if force_random_sample != -1:
            samples_weight = np.array([samples_weight[t] if i in idxs else 0 for i, t in enumerate(targets)])
        else:
            samples_weight = np.array([samples_weight[t] for t in targets])
            
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        if force_random_sample != -1:
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, fix_sample, replacement = True)
        else:
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, fix_sample, replacement = True)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers)
        
        if fix_sample_val == -1:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                num_workers=workers)
        else:
            targets = np.array(test_set.targets)
            samples_weight = np.array([1 for i in np.unique(targets)])
            samples_weight = np.array([samples_weight[t] for t in targets])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            val_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, fix_sample_val)
        
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=workers)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers)
    
        val_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers)

    unseen_test_loader = torch.utils.data.DataLoader(
        unseen_test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers)

    return train_loader, val_loader, unseen_test_loader