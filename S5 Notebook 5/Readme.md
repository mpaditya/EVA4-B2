This is the fifth notebook/experiment for the S5 Assignment Solution. In this notebook, we have added Max Pooling and (1X1) Conv Blocks to previous model.

### Target

Use Global Average Pooling (GAP) to convert 2D to 1D inputs.

Also, increase the capacity of the network (while keeping model size reasonable, ~ 10k) in order to improve training as well as test accuracy.

Finally, get rid of the big 5X5 kernel at the end & set bias as False in the Convolution Layers.

### Results

Total Parameters: 9910

Best Training Accuracy: 98.96

Best Test Accuracy: 99.40

### Analysis

Model has attained test acuracy of 99.40 there may be scope for improvement.

We can improve the test accuracy (reduce mis-classifications) by using Image Augmentation.the next Notebook.
