## Assignment S7

### Target 
Achieve 80% Accuracy on CIFAR-10 Dataset with less than 1M parameters with RF more than 44 using Dilated and Depthwise Convolutionas and GAP

### Results
**Parameters:** 332,736

**Best Test Accuracy:** 85.2% 

**Accuracy by Class:**

Accuracy of plane : 86 %

Accuracy of   car : 92 %

Accuracy of  bird : 78 %

Accuracy of   cat : 70 %

Accuracy of  deer : 74 %

Accuracy of   dog : 72 %

Accuracy of  frog : 80 %

Accuracy of horse : 80 %

Accuracy of  ship : 100 %

Accuracy of truck : 69 %

Custom Modules (.py files) were built and uploaded on to Google Drive. Drive was then mounted on Google Colab and these modules were imported.  

### Data Preproces Module

This module performs Data Augmentation Transforms on the train and test data sets and creates and returns data_loader objects.

### Model Module

This module contains the Model Architecture. The function get_model_instance takes the dropout value as input and initializes an object of class Net. 
The model includes 2 layers of Dilated Convolutions as well as 2 layers of Depthwise Separable Convolutions (which is incorporated using object of Class depthwise_separable_conv. 

### Train Test Module

This module defines functions for training and testing including forard pass, loss updation and backward pass. 

### Main File

The notebook "EVA4_S7_Aditya.ipynb" is the main file where the modules are imported and which contains the training and test logs.  
