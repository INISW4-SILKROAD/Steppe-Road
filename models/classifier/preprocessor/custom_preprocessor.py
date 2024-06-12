import torch.nn as nn

# 모델 정의 - 임의 길이의 벡터를 3*32*32로 변환
class CustomPreprosessor(nn.Module):
    def __init__(self, input = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, 1*32*32), 
            )
        self.processor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(3), 
            nn.ReLU()
            )
          
    def forward(self, x):
        x = self.fc(x)  
        x = x.view(-1, 1, 32, 32)
        x = self.processor(x)
        x = x.view(-1, 3, 32, 32) 
        return x
