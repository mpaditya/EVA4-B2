This is the first notebook/experiment for the S5 Assignment Solution. 
In this notebook, predominantly Convolution blocks have been used without any special layers to establish the baseline. 

### Target

Establish an error-free, working code.

Build the data pipeline from data ingestion, transforms, data loader and train and test loop functions.

Build a simple model focussing primarily on Convolution layers.

Do not worry about model size or overfitting. Try to fit the data well and achieve good training accuracy.

### Results

Total Parameters: 62894

Best Training Accuracy: 99.50

Best Test Accuracy: 99.01

### Analysis

The model fits the data well but it is too big (62k parameters)

The training accuracy outperforms test accuracy significantly indicating overfitting.

We will use Max Pooling and (1X1) Convolutions in next Notebook to ensure the most relevant features are carried forward in the network and the model is reasonable. This has 2 benefits:

Reduce Overfitting by not carrying forward noise deep in the network

Making the model lighter (lesser parameters)
