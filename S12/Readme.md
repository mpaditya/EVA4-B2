## Assignment S12A: EVA4_S12A_Aditya_2.ipynb

**Best Test Accuracy:** 78.26% (after 14 epochs)

**Training Set Transforms:** HorizontalFlip, Rotate, Cutout, HueSaturationValue, Normalize



## Assignment S12A: EVA4_S12B_Aditya.ipynb

50 Dog Images are in the folder "Dog_Images"

The json is "annotations_dogs.json"

The json contains the annotations for the 50 dog images. An example of the fields for the sample image is as shown below:

{'area': 124166,
 'bbox': [23, 4, 343, 362],
 'category_id': 1,
 'id': 0,
 'image_id': '0',
 'iscrowd': 0,
 'segmentation': [23, 4, 366, 4, 366, 366, 23, 366]}

The optimal number of Clusters (Anchor Boxes) for the scaled bounding Boxes of these 50 images was 4. 

