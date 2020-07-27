import os
import torch
import torchvision


SEED = 1

class Data():

  def getTrainDataSet(self):
    datadir = os.getcwd()+'/MergeData/Train'
    # datadir = os.getcwd()+'/tiny-imagenet-200/train'
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=data_preprocess_Albumentation_module_1.albumentation_train())
    num_train = len(dataset)
    print("Train Data Size : ", num_train)
    return dataset

  def getTestDataSet(self):
    datadir = os.getcwd()+'/MergeData/Val'
    # datadir = os.getcwd()+'/tiny-imagenet-200/val'
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=data_preprocess_Albumentation_module_1.albumentation_test())
    num_train = len(dataset)
    print("Train Data Size : ", num_train)
    return dataset

  def get_tiny_imagenet_train_dataset(train_transforms, train_image_data, train_image_labels):
    from tinyimagenetdataset import TinyImagenetDataset
    return TinyImagenetDataset(image_data=train_image_data, image_labels=train_image_labels,
                               transform=data_preprocess_Albumentation_module_1.albumentation_train())
     

  def get_tiny_imagenet_test_dataset(test_transforms, test_image_data, test_image_labels):
    from tinyimagenetdataset import TinyImagenetDataset
    return TinyImagenetDataset(image_data=test_image_data, image_labels=test_image_labels, 
                               transform=data_preprocess_Albumentation_module_1.albumentation_test())
    