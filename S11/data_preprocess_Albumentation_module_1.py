import torch, torchvision
from torchvision import datasets, transforms
from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

# custom dataset class for albumentations library
class albumentation_train:
    def __init__(self):
        self.albumentation_transform = Compose([HorizontalFlip(),
                                                Rotate(limit=2),
                                                HueSaturationValue(hue_shift_limit=3, sat_shift_limit=2),
                                                # RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                                                PadIfNeeded(min_height = 40, min_width = 40,border_mode=4, value=None, p=1),
                                                RandomCrop(32,32),
                                                Normalize(
                                                    mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5],
                                                ),
                                                Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
                                                ToTensor()
                                                ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transform(image=img)
        return img['image']


class albumentation_test:
    def __init__(self):
        self.albumentation_transform = Compose([
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transform(image=img)
        return img['image']
    
SEED = 1
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    
    
train_dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, num_workers = 2, batch_size=64)
test_dataloader_args = dict(shuffle=False, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=False, num_workers = 2, batch_size=64)

# train dataloader

def get_train_test_loader(tr,ts):
    train_loader = torch.utils.data.DataLoader(tr, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(ts, **test_dataloader_args)
    return train_loader, test_loader

