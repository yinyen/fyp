from pytorch.xception_block import SeparableConv2d, Block, init_all
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewSmallXception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_ftrs = 1000, num_classes=5):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(NewSmallXception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(inplace=True)


        # last = 1024
        last = 728
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.mp1 = nn.MaxPool2d(3, stride=1)
        
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(728,last,2,2,start_with_relu=True,grow_first=False)
        self.mp2 = nn.MaxPool2d(3, stride=1)

        self.conv3 = SeparableConv2d(last,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)


        self.fc = nn.Linear(2048, num_ftrs)

        self.fc2 = nn.Linear(num_ftrs*2, 2048)
        self.relu21 = nn.LeakyReLU(inplace=True)
        self.fc3 = nn.Linear(2048, 512)
        self.relu31 = nn.LeakyReLU(inplace=True)

        self.fc_out = nn.Linear(512, num_classes)
      
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.mp1(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input, input2):
        x = self.features(input)
        x = self.logits(x) # eye1 features
        x2 = self.features(input2)
        x2 = self.logits(x2) # eye2 features
        x_list = [x, x2]
        x = torch.cat(x_list, dim = 1)

        x = self.fc2(x)
        x = self.relu21(x)
        x = self.fc3(x)
        x = self.relu31(x)
        x = self.fc_out(x)
        return x


def new_small_dual_xception(num_ftrs = 1000, num_classes=5, mean = 0, std = 0.1, use_init = "normal", limit_two = 1):
    model = NewSmallXception(num_ftrs = num_ftrs, num_classes=num_classes)
    if use_init == "normal":
        init_all(model, torch.nn.init.normal_, limit_two = limit_two, mean=0., std=0.05) 
    elif use_init == "orthogonal":
        init_all(model, nn.init.orthogonal_, limit_two = limit_two) 
    elif use_init == "kaiming":
        init_all(model, nn.init.kaiming_uniform_, a = 0.01, limit_two = limit_two)
    return model
