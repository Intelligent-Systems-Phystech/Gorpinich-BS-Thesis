# This code is a modified copy of https://raw.githubusercontent.com/passalis/pkth/master/loaders/cifar_dataset.py
# We don't use suffling of train dataset (in this modification)
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def cifar10_loader(data_path='../data', batch_size=128, split_train_val=False,  maxsize=-1, use_aug = False):
    """
    Loads the cifar10 dataset in torch-ready format
    :param data_path:
    :param batch_size:
    :return:
    """

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    if use_aug:
        train_data_original = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)
    else:    
        train_data_original = dset.CIFAR10(data_path, train=True, transform=test_transform, download=True)
        
    if maxsize > 0:
        train_data_original = torch.utils.data.Subset(train_data_original, list(range(maxsize)))  
       
    if split_train_val:
        if maxsize > 0:
            valid_offset = maxsize - maxsize//10
            valid_data_original = torch.utils.data.Subset(train_data_original, list(range(valid_offset, maxsize)))
            train_data_original = torch.utils.data.Subset(train_data_original, list(range(valid_offset)))
        else:
            valid_data_original = torch.utils.data.Subset(train_data_original, list(range(45000, 50000)))
            train_data_original = torch.utils.data.Subset(train_data_original, list(range(45000)))  
     
    test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                              pin_memory=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch_size,
                                                        shuffle=False, num_workers=0, pin_memory=True)
    if  split_train_val:
      valid_loader_original = torch.utils.data.DataLoader(valid_data_original, batch_size=batch_size,
                                            shuffle=False, num_workers=0, pin_memory=True)   
      return train_loader_original, valid_loader_original, test_loader

    return train_loader_original, test_loader
