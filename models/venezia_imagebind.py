'''
venezia_imagebind.py
촉감 추정 모델 - imagebind 기반
혼용률과 이미지 피처 벡터를 결합할 때 단점 극복
bilinear polling을 이용하여 균등한 결합 시도

class:
    Venezia:
        Bilinear polling으로 결합 
        결합 이후 mobile_net으로 분류
        + 레이어 정보
            + image_encoder 
                + custum_ibvis_encoder - img > 1*512 피처벡터
                + 기존 imagebind 가중치 사용
                + 마지막 레이어 학습
            + polling: 
                + bilinear polling - 1*512 + 1*12 > 1*512
                + 학습
            + midprocess: 
                + 1*512 > fc > conv.2d > batch.norm2d > ReLU > 3*32*32
                + 학습
            + classifier
                + custom_mobilenet - batch*3*512*512 > batch*5
                + 기존 mobilenet_v3_small 가중치 사용
                + 분류 레이어만 5 class로 줄여 학습 
        + 기존 imagebind 가중치를 가져옴
        + midprocess, polling, classifier_ 학습함

작성자: 윤성진
'''

import torch.nn as nn

# 자체 라이브러리
import models.encoder.custom_ibvis_encoder as cibv
from models.classifier.custom_mobile_net import CustomMobileNet

class Venezia(nn.Module):
    def __init__(self, vision_embed_dim = 512, portion_dim = 12, output_dim = 512):
        super(Venezia, self).__init__()
        self.image_encoder = cibv.CustomIbvisEncoder(out_embed_dim=vision_embed_dim)
        self.polling = nn.Bilinear(vision_embed_dim, portion_dim, output_dim)
        self.linear = nn.Linear(output_dim, 1*32*32)
        self.midprocessor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(3), 
            nn.ReLU()
        )

        self.classifier_ = CustomMobileNet(5)
        
    def forward(self, x_1, x_2):
        # 이미지 인코딩
        x_1 = self.image_encoder(x_1)
        
        # 피처 결합
        x = self.polling(x_1, x_2)
        batch_size = x.size(0)
        x = self.linear(x)
        x = x.view(batch_size, 1, 32, 32)
        # 이미지로 변환
        x = self.midprocessor(x)

        # 분류
        result = self.classifier_(x)
        return result
