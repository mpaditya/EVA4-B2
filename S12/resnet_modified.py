import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.d1 = nn.Dropout2d(0.0)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.d1(F.relu(self.bn1(self.conv1(x))))
        x = (F.relu(self.bn2(self.conv2(x))))
        return x


class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1, self).__init__()
        self.in_planes = 128
        self.d = nn.Dropout2d(0.0)
        self.channels = [64, 128, 256, 512]

        #INPUT
        self.conv0 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        
        #LAYER 1
        self.conv1 = nn.Conv2d(self.channels[0],self.channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(self.channels[1])
        self.layer1 = self._make_layer(block, self.channels[1], num_blocks[0], stride=1)

        #LAYER 2
        self.conv2 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(self.channels[2])

        #LAYER 3
        self.conv3 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(self.channels[3])
        self.layer3 = self._make_layer(block, self.channels[3], num_blocks[1], stride=1)

        self.mp1 = nn.MaxPool2d(4)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.d(F.relu(self.bn1(self.conv0(x))))
        out = self.d(F.relu(self.bn2(self.mp(self.conv1(out)))))
        out = out + self.layer1(out)
        out = self.d(F.relu(self.bn3(self.mp(self.conv2(out)))))
        out = self.d(F.relu(self.bn4(self.mp(self.conv3(out)))))
        out = out + self.layer3(out)
        out = self.mp1(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_mod():
    return ResNet1(BasicBlock, [1,1])