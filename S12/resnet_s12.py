import torch
import torch.nn as nn
import torch.nn.functional as F




class Net (nn.Module):
    def __init__(self):
        super (Net, self).__init__()
        
        #prep layer
        self.prep= self.prep_block(3, 64, 3, 1) #input layer, o/p = 32x32x64
        
        #Layer 1
        self.conv1 = self.conv_block(64, 128, 3, 1, 2) #o/p = 16x16x128
        self.res1 = self.res_block(128, 128, 3, 1)

        #Layer 2
        self.conv2 = self.conv_block(128, 256, 3, 1, 2)  #o/p = 8x8x256

        #Layer 3 
        self.conv3 = self.conv_block(256, 512, 3, 1, 2) #o/p = 4x4x512
        self.res3 = self.res_block(512, 512, 3, 1)

        self.pool1 = nn.MaxPool2d(4) #o/p = 1x1x512
        #self.conv4 = nn.Conv2d(512,10,1)
        self.linear = nn.Linear(512,10) # 512x10
        
    def prep_block(self,inputs, output, kernel, p):
        prep_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                 nn.BatchNorm2d(output),
                                 nn.ReLU())
        return prep_bloc

    def conv_block(self, inputs, output, kernel, p, m):
        conv_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                 nn.MaxPool2d(m),
                                 nn.BatchNorm2d(output),
                                 nn.ReLU())
        return conv_bloc

    def res_block(self, inputs, output, kernel, p):
        res_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                nn.BatchNorm2d(output),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                nn.BatchNorm2d(output),
                                nn.ReLU()
                                )
        return res_bloc

        
    def forward(self, x):
        x = self.prep(x) #i/p
        x = self.conv1(x) 
        r1 = self.res1(x)
        x1 = x + r1 
        x = x + F.relu(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        r3 = self.res3(x)
        x1 = x + r3
        x = x + F.relu(x1)
        x = self.pool1(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        x = F.softmax(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
       # self.dp1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = self.dp1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())