import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===========================
# 1. 定义 SimpleCNN (骨干网络1)
# ===========================
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=128, dropout_rate=0.4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ===========================
# 2. 定义 ResNet18 (骨干网络2)
# ===========================
class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        # 使用 verify=False 或 weights 参数根据 torchvision 版本调整
        # 这里为了兼容性使用 weights
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except:
            # 旧版本 torchvision 兼容
            resnet = models.resnet18(pretrained=pretrained)

        # 修改第一层卷积以接受单通道灰度图 (1 channel)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 如果使用预训练权重，将RGB权重的平均值赋给单通道
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===========================
# 3. 定义主模型 EmbeddingModel
# (你的报错是因为缺了这个类)
# ===========================
class EmbeddingModel(nn.Module):
    def __init__(self, backbone='simple_cnn', embedding_dim=128, pretrained=True):
        super(EmbeddingModel, self).__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        if backbone == 'simple_cnn':
            self.backbone = SimpleCNN(embedding_dim=embedding_dim)
        elif backbone == 'resnet18':
            self.backbone = ResNet18Backbone(embedding_dim=embedding_dim, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)