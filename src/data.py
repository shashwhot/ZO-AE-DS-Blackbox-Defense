import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(dataset_name='cifar10', batch_size=256, workers=4):
    """
    Creates Training and Testing DataLoaders.
    """
    # 1. Image Transformations
    # Convert raw pixels (0-255) into PyTorch tensors (0.0 - 1.0)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 2. Select and Download the Dataset
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Dataset not supported. Choose 'cifar10' or 'mnist'.")

    # 3. Create the Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, test_loader