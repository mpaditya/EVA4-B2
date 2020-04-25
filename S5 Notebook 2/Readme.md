This is the second notebook/experiment for the S5 Assignment Solution. 
In this notebook, we have added Max Pooling and (1X1) Conv Blocks to previous model.  

### Target

Use Max Pooling and (1X1) Convolutions to make model lighter and reduce overfitting. Trim 1 or 2 Convolution Layers if necessary.

Retain the basic structure/architecture of the model as far as possible.

### Results

Total Parameters: 9702

Best Training Accuracy: 99.34

Best Test Accuracy: 98.82

### Analysis

The model is still overfitting the data to some extent.

The model is light (only 9k parameters) but training accuracy has reduced because of reducing the capacity of the model.

In the next Notebook, we will use Batch Normalization (BN) to increase the efficiency of back propagation by ensuring that inputs are normalized (mean ) ans std dev 1) before feeding to each layer.
