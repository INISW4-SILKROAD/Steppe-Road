'''
constantinople.py
촉감 추정 모델
기존 비전인코더+혼용률 벡터 -> 분류모델 구조 한계 극복 
오토 인코더, Bilinear polling 등 결합

class:
    Constantinople:
        인코딩된 정보를 Bilinear polling으로 결합 후 디코딩
        + 레이어 정보
            + image_encoder 
                + custum_ibvis_encoder - img > 1*512 피처벡터
                + 기존 imagebind 가중치 사용
                + 최종 레이어 512로 줄여 학습
            + portion_encoder:
                + SimpleAE - 1*4 > 1*512
                + 자체 학습시킨 가중치 사용
            + polling: 
                + bilinear polling - 1*512 + 1*512 > 1*512
                + 학습
            + 촉감 디코더: 
                + SimpleAE - 1*512 > 1*4
                + 자체 학습시킨 가중치 사용 
        + 기존 constantinople 가중치를 가져옴
        + midprocess, polling 학습함
        + 인코더와 디코더 모두 미리 학습시켜 사용
        + polling 레이어만 학습함

작성자: 윤성진
'''

import torch.nn as nn

# 자체 라이브러리
import models.encoder.simple_ae as cae
import models.encoder.custom_ibvis_encoder as cibv


class Constantinople(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, touch_dim = 4):
        super(Constantinople, self).__init__()
        self.image_encoder = cibv.CustomIbvisEncoder(out_embed_dim=latent_dim)
        self.portion_encoder = cae.SimpleAE(
            input_dim=portion_dim, 
            latent_dim=latent_dim
            ).encoder
        
        self.polling = nn.Bilinear(
            in1_features=latent_dim, 
            in2_features=latent_dim, 
            out_features=latent_dim
            )
        
        self.touch_decoder = cae.SimpleAE(
            input_dim=touch_dim,
            latent_dim=latent_dim
            ).decoder
        
        
    def forward(self, vision, portion):
        # 각각 인코딩
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        
        # Bilinear polling으로 피처 벡터 결함(잠재 공간 결합)
        latent = self.polling(vision, portion)
        
        # 촉감 정보 디코딩
        result = self.touch_decoder(latent)
        return result