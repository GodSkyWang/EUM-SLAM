import torch
import torch.nn as nn
import torch.nn.functional as F

# class BottleneckBlock(nn.Module):
#     def __init__(self, in_planes, planes, norm_fn='group', stride=1):
#         super(BottleneckBlock, self).__init__()
  
#         self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
#         self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)

#         num_groups = planes // 8

#         if norm_fn == 'group':
#             self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
#             self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
#             self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
#             if not stride == 1:
#                 self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
#         elif norm_fn == 'batch':
#             self.norm1 = nn.BatchNorm2d(planes//4)
#             self.norm2 = nn.BatchNorm2d(planes//4)
#             self.norm3 = nn.BatchNorm2d(planes)
#             if not stride == 1:
#                 self.norm4 = nn.BatchNorm2d(planes)
        
#         elif norm_fn == 'instance':
#             self.norm1 = nn.InstanceNorm2d(planes//4)
#             self.norm2 = nn.InstanceNorm2d(planes//4)
#             self.norm3 = nn.InstanceNorm2d(planes)
#             if not stride == 1:
#                 self.norm4 = nn.InstanceNorm2d(planes)

#         elif norm_fn == 'none':
#             self.norm1 = nn.Sequential()
#             self.norm2 = nn.Sequential()
#             self.norm3 = nn.Sequential()
#             if not stride == 1:
#                 self.norm4 = nn.Sequential()

#         if stride == 1:
#             self.downsample = None
        
#         else:    
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

#     def forward(self, x):
#         y = x
#         y = self.relu(self.norm1(self.conv1(y)))
#         y = self.relu(self.norm2(self.conv2(y)))
#         y = self.relu(self.norm3(self.conv3(y)))

#         if self.downsample is not None:
#             x = self.downsample(x)

#         return self.relu(x+y)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        # 定义两个卷积层，用于在残差模块中进行特征提取
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True) # 定义ReLU激活函数，用于增加模型的非线性

        num_groups = planes // 8  # 根据输入的norm_fn参数选择不同的标准化层

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            #GroupNorm是一种标准化层，它可以在深度学习中用于减少内部协变量偏移。
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            #InstanceNorm2d是一种标准化层，它可以在深度学习中用于减少内部协变量偏移。
            #InstanceNorm2d对每个实例的所有通道进行标准化，而GroupNorm对每个组的所有实例进行标准化。
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


DIM=32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)
#BatchNorm2d层是神经网络中的一种归一化层，它可以在批处理内进行归一化，通常用于图像处理任务。
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)
#InstanceNorm2d层是神经网络中的一种归一化层，它可以在每个样本上进行归一化，通常用于图像处理任务。
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules(): # 遍历模型的所有模块
            if isinstance(m, nn.Conv2d): # 如果模块是2D卷积层
                # 使用Kaiming正常分布初始化卷积层的权重
                # mode='fan_out' 表示按照fan_out进行初始化
                # nonlinearity='relu' 表示在ReLU激活函数下进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            # 如果模块是批量归一化层、实例归一化层或组归一化层    
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)): 
                if m.weight is not None: # 如果模块有权重（即，gamma）
                    nn.init.constant_(m.weight, 1) # 将权重初始化为1
                if m.bias is not None: # 如果模块有偏置（即，beta）
                    nn.init.constant_(m.bias, 0) # 将偏置初始化为0

    def _make_layer(self, dim, stride=1): # 创建两个残差块
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)  # 将两个残差块打包成序列
        
        self.in_planes = dim # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape 
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)

# class BasicEncoder4(nn.Module):
#     def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
#         super(BasicEncoder4, self).__init__()
#         self.norm_fn = norm_fn
#         self.multidim = multidim

#         if self.norm_fn == 'group':
#             self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
#         elif self.norm_fn == 'batch':
#             self.norm1 = nn.BatchNorm2d(DIM)

#         elif self.norm_fn == 'instance':
#             self.norm1 = nn.InstanceNorm2d(DIM)

#         elif self.norm_fn == 'none':
#             self.norm1 = nn.Sequential()

#         self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.in_planes = DIM
#         self.layer1 = self._make_layer(DIM,  stride=1)
#         self.layer2 = self._make_layer(2*DIM, stride=2)

#         # output convolution
#         self.conv2 = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

#         if dropout > 0:
#             self.dropout = nn.Dropout2d(p=dropout)
#         else:
#             self.dropout = None

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
#                 if m.weight is not None:
#                     nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def _make_layer(self, dim, stride=1):
#         layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
#         layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
#         layers = (layer1, layer2)
        
#         self.in_planes = dim
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         b, n, c1, h1, w1 = x.shape
#         x = x.view(b*n, c1, h1, w1)

#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = self.conv2(x)

#         _, c2, h2, w2 = x.shape
#         return x.view(b, n, c2, h2, w2)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockWithSE(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlockWithSE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        
        # 根据norm_fn选择标准化方法
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(planes // 8, planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        
        if norm_fn == 'group':
            self.norm2 = nn.GroupNorm(planes // 8, planes)
        elif norm_fn == 'batch':
            self.norm2 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm2 = nn.InstanceNorm2d(planes)
        else:
            self.norm2 = nn.Sequential()
        
        # Squeeze-and-Excitation layer
        self.se = SELayer(planes)
        
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm1 if norm_fn != 'none' else nn.Sequential()
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class BasicEncoder1(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder1, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)
#BatchNorm2d层是神经网络中的一种归一化层，它可以在批处理内进行归一化，通常用于图像处理任务。
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)
#InstanceNorm2d层是神经网络中的一种归一化层，它可以在每个样本上进行归一化，通常用于图像处理任务。
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)
                # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules(): # 遍历模型的所有模块
            if isinstance(m, nn.Conv2d): # 如果模块是2D卷积层
                # 使用Kaiming正常分布初始化卷积层的权重
                # mode='fan_out' 表示按照fan_out进行初始化
                # nonlinearity='relu' 表示在ReLU激活函数下进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            # 如果模块是批量归一化层、实例归一化层或组归一化层    
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)): 
                if m.weight is not None: # 如果模块有权重（即，gamma）
                    nn.init.constant_(m.weight, 1) # 将权重初始化为1
                if m.bias is not None: # 如果模块有偏置（即，beta）
                    nn.init.constant_(m.bias, 0) # 将偏置初始化为0
    
    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlockWithSE(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlockWithSE(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)
        
    def forward(self, x):
        b, n, c1, h1, w1 = x.shape 
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)