This is the second notebook/experiment for the S5 Assignment Solution. In this notebook, we have added Max Pooling and (1X1) Conv Blocks to previous model.

### Target

Apply Batch Normalization (BN) to every Convolution Layer and check the results.

### Results

Total Parameters: 9894

Best Training Accuracy: 99.65

Best Test Accuracy: 99.26

### Analysis

The model is overfitting.

The training acuracy is 99.65%. So this means that we should not increase the capacity of this model in the immediate subsequent step. The only way to boost the test accuracy is by countering overfitting whilst keeping model capacity in check.

In the next Notebook, we will use Dropout of 5%, 10% and 20% (and select best of these 3 variants) and check model's performance.
