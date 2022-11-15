import os
import torch
from torchvision import datasets, transforms

def create_dataloaders(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir) or not os.path.exists(test_dir):
        raise Exception("Invalid data directory.")
    
    batch_size = 64
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir , transform=train_transforms) 
    valid_datasets = datasets.ImageFolder(valid_dir , transform=test_transforms) 
    test_datasets = datasets.ImageFolder(test_dir , transform=test_transforms) 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)
    
    class_to_idx = train_datasets.class_to_idx
    
    return class_to_idx, train_dataloaders, valid_dataloaders, test_dataloaders