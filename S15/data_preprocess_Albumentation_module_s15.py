import torch, torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

# custom dataset class for albumentations library
class albumentation_dataset:
    def __init__(self):
        self.albumentation_transform = transforms.Compose([transforms.Resize((192,192)),transforms.ToTensor()])

    def getTransform(self):
        return self.albumentation_transform
    
SEED = 1
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    
    
train_dataloader_args = dict(shuffle=True, batch_size=32, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, num_workers = 2, batch_size=64)
test_dataloader_args = dict(shuffle=False, batch_size=32, num_workers=4, pin_memory=True) if cuda else dict(shuffle=False, num_workers = 2, batch_size=64)

# train dataloader

def get_train_test_loader(tr,ts):
    train_loader = torch.utils.data.DataLoader(tr, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(ts, **test_dataloader_args)
    return train_loader, test_loader

