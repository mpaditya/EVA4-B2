This is the fourth notebook/experiment for the S5 Assignment Solution. In this notebook, we have added Max Pooling and (1X1) Conv Blocks to previous model.

### Target

Use Dropout of 5%, 10%, 15% and 20% and select best of these 3 variants and report model's performance.

### Results

Total Parameters: 9894

Best Training Accuracy: 99.24

Best Test Accuracy: 99.36

### Analysis

The model is not overfitting anymore. In fact, test accuracy is marginally better than training accuracy.

However, both training as well as test accuracy can be improved. But we need to do this without increasing number of parameters too much.

This is where GAP (Global Average Pooling) comes into play. This will be explored in the next Notebook.
