import torch.nn as nn
import torchvision.models as models

# 모델 정의
class CustomMobileNet(nn.Module):
    def __init__(self, num_class = 5):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_class)
        
    def forward(self, x):
        x = self.mobilenet(x)
        return x
