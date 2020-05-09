import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self, drop_val = 0.05):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            
            # CONVOLUTION BLOCK C1
            Conv2d(3, 32, kernel_size=3, padding=1), # RF 3, output_size = 32
            BatchNorm2d(32),           
            ReLU(inplace=True),
            nn.Dropout(drop_val),            

            Conv2d(32, 64, kernel_size=3, padding=1), # RF 5, output_size = 32
            BatchNorm2d(64),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),            

            # DILATED CONVOLTION LAYER BELOW
            Conv2d(64, 128, kernel_size=3, padding=2, dilation = 2), # RF 9, output_size = 32
            BatchNorm2d(128),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),                  


            # TRANSITION BLOCK 1
            MaxPool2d(kernel_size=2), # RF 8, output_size = 16
            
            Conv2d(128, 32, kernel_size=1, padding=0),  # RF 8, output_size = 16
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),            


            # CONVOLUTION BLOCK C2          
            Conv2d(32, 32, kernel_size=3, padding=1), # RF 12, output_size = 16
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),            

            Conv2d(32, 64, kernel_size=3, padding=1), # RF 16, output_size = 16
            BatchNorm2d(64),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),                  

            # DILATED CONVOLTION LAYER BELOW
            Conv2d(64, 128, kernel_size=3, padding=2, dilation = 2), # RF 34, output_size = 16
            BatchNorm2d(128),             
            ReLU(inplace=True),
            nn.Dropout(drop_val), 


            # TRANSITION BLOCK 2
            MaxPool2d(kernel_size=2), # RF 18, output_size = 8
            
            Conv2d(128, 32, kernel_size=1, padding=0),  # RF 18, output_size = 8
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val), 

            # CONVOLUTION BLOCK C3          
            Conv2d(32, 32, kernel_size=3, padding=1), # RF 42, output_size = 8
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),            

            # DEPTHWISE SEPARABLE CONVOLUTION
            depthwise_separable_conv(32,64), # RF 26, output_size = 8
            BatchNorm2d(64),             
            ReLU(inplace=True),
            nn.Dropout(drop_val), 

            # DEPTHWISE SEPARABLE CONVOLUTION
            depthwise_separable_conv(64,128), # RF 26, output_size = 8
            BatchNorm2d(128),             
            ReLU(inplace=True),
            nn.Dropout(drop_val), 


            # TRANSITION BLOCK 3
            MaxPool2d(kernel_size=2), # RF 3, output_size = 4
            
            Conv2d(128, 32, kernel_size=1, padding=0),  # RF 34, output_size = 4
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val), 

            # CONVOLUTION BLOCK C4        
            Conv2d(32, 32, kernel_size=3, padding=1), # RF 42, output_size = 4
            BatchNorm2d(32),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),            

            Conv2d(32, 64, kernel_size=3, padding=1), # RF 50, output_size = 4
            BatchNorm2d(64),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),

            Conv2d(64, 128, kernel_size=3, padding=1), # RF 50, output_size = 4
            BatchNorm2d(128),             
            ReLU(inplace=True),
            nn.Dropout(drop_val),

            # GAP
            nn.AdaptiveAvgPool2d((1,1)), # RF 3, output_size = 1
            Conv2d(128, 10, kernel_size=1, bias = False) # RF 3, output_size = 1

        )



    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


def get_model_instance(drop_val):
	return Net(drop_val = drop_val)