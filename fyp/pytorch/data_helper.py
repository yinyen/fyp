import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def initialize_dataset(main_data_dir = "./data3", batch_size = 64, size = 100, workers = 16):
    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(root=f'{main_data_dir}/train', transform=transform_train)
    test_set = datasets.ImageFolder(root=f'{main_data_dir}/val', transform=transform_test)
    unseen_test_set = datasets.ImageFolder(root=f'{main_data_dir}/test', transform=transform_test)

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