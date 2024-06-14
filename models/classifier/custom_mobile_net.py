"""
custom_mobilnet.py
모바일넷의 최종 분류 레이어를 쉽게 조절할 수 있도록 조정

class:
    CustomMibileNet: 
        mobilenet_v3_small의 가중치를 로드하고, 
        분류기의 마지막 레이어를 num_class 만큼 분류할 수 있도록 조정
         
작성자: 윤성진
"""
import torch.nn as nn
import torchvision.models as models

class CustomMobileNet(nn.Module):
    '''
    분류 클래스를 변형시킨 mobilenet_v3_small 모듈
    객체 생성시 받은 num_class으로 최종 레이어 변환 
    
    method:
        foward: 동작 방식 정의
    '''
    def __init__(self, num_class = 5):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_class)
        
    def forward(self, x):
        x = self.mobilenet(x)
        return x
