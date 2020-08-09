# -*- coding: utf-8 -*-

import torch
import torchvision
import data_preprocess_Albumentation_module_s15 as alb
SEED = 1

### Defining dataset
from pathlib import Path
from PIL import Image


class DataSet():

  def __init__(self, data_root):
    alb_obj = alb.albumentation_dataset()
    self.albumentation_transform = alb_obj.getTransform()
    f1 = Path(data_root+'bg_192/')
    self.f1_files = list(sorted(f1.glob('*.jpg')))

    f2 = Path(data_root+'fg_bg/')
    self.f2_files_ = list(sorted(f2.glob('*.jpg')))

    f3 = Path(data_root+'mask/')
    self.f3_files_ = list(sorted(f3.glob('*.jpg')))

#    f4 = Path(data_root+'fg_bg_depth_full/')
#    self.f4_files_ = list(sorted(f4.glob('*.jpg')))


    self.name = {}
    for i in self.f1_files:
      num = str(i).split('.')[0].split('bg192_')[1]
      self.name[num] = i
  
  #Extracting the number of the background_foreground, so that the same can be called for background
  def get_f1_image(self, index):
    a = str(self.f2_files_[index]).split('bg192_')[1].split('_fg')[0]
    return self.name[a]

  def __len__(self):
    return len((self.f2_files_))

  def __getitem__(self, index):
    f1_image_ = self.get_f1_image(index) #calling the background based on the index of the bg_fg
    f1_image = Image.open(f1_image_)
    f1_image = f1_image.convert(mode='RGB')
    f2_image = Image.open(self.f2_files_[index])
    f2_image = f2_image.convert(mode='RGB')
    f3_image = Image.open(self.f3_files_[index])
    f3_image = f3_image.convert(mode='L')
#    f4_image = Image.open(self.f4_files_[index])
#    f4_image = f4_image.convert(mode='L')
    
    
    f1_image = self.albumentation_transform(f1_image)
    f2_image = self.albumentation_transform(f2_image)
    f3_image = self.albumentation_transform(f3_image)
#    f4_image = self.albumentation_transform(f4_image)


    return {'f1': f1_image, 'f2' : f2_image, 'f3' : f3_image} 