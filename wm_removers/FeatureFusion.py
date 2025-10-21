import torch
import torch.nn as nn

# 定义特征融合模型
class FeatureFusionModel(nn.Module):
    def __init__(self):
        super(FeatureFusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)   # 1*1的卷积实现将通道数调整为与输入特征相同
        self.relu = nn.ReLU()

    def forward(self, synthesized, synthesized2):
        fused_features = self.conv1(synthesized) + self.conv1(synthesized2)
        fused_features = self.relu(fused_features)
        fused_features = self.conv2(fused_features)
        return fused_features