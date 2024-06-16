'''
constantinople_clip.py
촉감 추정 모델
기존 constantinople의 clip인코더 버전

class:
    ConstantinopleClip:
        인코딩된 정보를 Bilinear polling으로 결합 후 디코딩
        + 레이어 정보
            + image_encoder 
                + clip - img > 1*512 피처벡터
                + 기존 clip 가중치 사용  
            + portion_encoder:
                + SimpleAE - 1*4 > 1*512
                + 자체 학습시킨 가중치 사용
            + polling: 
                + 1*512 + 1*512 > bilinear polling > 1*512
                + 학습
            + 촉감 디코더: 
                + SimpleAE - 1*512 > 1*4
                + 자체 학습시킨 가중치 사용 
        + 인코더와 디코더 모두 미리 학습시켜 사용
        + polling 레이어만 학습함

작성자: 윤성진
'''

import torch.nn as nn

# clip - 다운받아야함
import clip

# 자체 라이브러리
import models.encoder.simple_ae as cae

class ConstantinopleClip(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, touch_dim = 4, device = 'cpu'):
        super(ConstantinopleClip, self).__init__()
        
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
        
        self.touch_decoder = cae.SimpleAE(
            input_dim=touch_dim,
            latent_dim=latent_dim
            ).decoder
        
        
    def forward(self, vision, portion):
        # 각각 인코딩
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        
        # 잠재 공간 결합 후 안정화
        latent = self.polling(vision, portion)
        latent = self.postprocess(latent)
        
        result = self.touch_decoder(latent)
        return result