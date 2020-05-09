import torch, torchvision
from torchvision import datasets, transforms


def get_train_transform(mu,sig):
  return transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     transforms.RandomRotation((-5.0, 5.0)),
     #transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
     #transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.ToTensor(),
     transforms.Normalize(mu, sig)])
  
def get_test_transform(mu,sig):
  return transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     #transforms.RandomRotation((-5.0, 5.0)),
     #transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
     #transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.ToTensor(),
     transforms.Normalize(mu, sig)])
   

SEED = 1
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    
    
train_dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, num_workers = 2, batch_size=64)
test_dataloader_args = dict(shuffle=False, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=False, num_workers = 2, batch_size=64)

# train dataloader

def get_train_test_loader(tr,ts):
    train_loader = torch.utils.data.DataLoader(tr, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(ts, **test_dataloader_args)
    return train_loader, test_loader
