'''
constantinople_v2.py
촉감 추정 모델
기존 constantinople의 불안정성 개선 
결합된 latent space에 대해 midprocess로 안정화 

class:
    ConstantinopleV2:
        인코딩된 정보를 Bilinear polling으로 결합 
        결합 이후 midprocess로 latentspace안정화
        + 레이어 정보
            + image_encoder 
                + custum_ibvis_encoder - img > 1*512 피처벡터
                + 기존 constantinople 가중치 사용
            + portion_encoder:
                + SimpleAE - 1*4 > 1*512
                + 자체 학습시킨 가중치 사용
            + polling: 
                + bilinear polling - 1*512 + 1*512 >  1*512
                + 학습
            + midprocess: 
                + 1*512 > dropout > fc > batch.norm > ReLU > 1*512
                + 학습
            + 촉감 디코더: 
                + SimpleAE - 1*512 > 1*4
                + 자체 학습시킨 가중치 사용 
        + 기존 constantinople 가중치를 가져옴
        + midprocess, polling 학습함

작성자: 윤성진
'''

import torch.nn as nn

# clip
import clip

# 자체 라이브러리
import models.encoder.simple_ae as cae


class ConstantinopleV2(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, touch_dim = 4, device = 'cpu'):
        super(ConstantinopleV2, self).__init__()
        clip_encoder, _ = clip.load("ViT-B/32", device=device)
        
        self.image_encoder = clip_encoder.encode_image        
        self.portion_encoder = cae.SimpleAE(
            input_dim=portion_dim, 
            latent_dim=latent_dim
            ).encoder
        
        self.polling = nn.Bilinear(
            in1_features=latent_dim, 
            in2_features=latent_dim, 
            out_features=latent_dim
            )
        
        self.midprocess = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(latent_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()  
            )        

        self.touch_decoder = cae.SimpleAE(
            input_dim=touch_dim,
            latent_dim=latent_dim
            ).decoder
        
        
    def forward(self, vision, portion):
        # 각각 인코딩
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        
        # 잠재공간 결합 및 안정화 
        latent = self.polling(vision.float(), portion)
        latent = self.midprocess(latent)
        
        # 촉감 디코딩
        result = self.touch_decoder(latent)
        return result