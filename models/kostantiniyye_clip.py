'''
kostatiniyye_clip.py
촉감 추정 모델
기존 kostatiniyye 모델의 과적합 문제 해결 시도
드롭아웃을 어텐션 양쪽에 추가하여 과적합 방지

class:
    Kostantiniyye:
        혼용률 인코더에 정규화 레이어 추가
        인코딩된 정보를 행렬곱으로 결합 하여 피처 맵 생성
        + 각 촉감에 대해 별개의 모델을 적용함
            + 피처맵에 어텐션 메커니즘을 적용
            + 모바일넷에 적용
        + 레이어 정보
            + image_encoder 
                + clip - batch*img > batch*512
                + 기존 clip 가중치 사용  
                + ***중요 출력 float 아님 - 반드시 vision에 float변환 해줄 것
            + portion_encoder:
                + SimpleAE - batch*4 > batch*512
                + 자체 학습시킨 가중치 사용
            + encoder_normalize:
                + layerNorm - batch*512 > batch*512
                + 학습
            + 행렬곱
                + 행렬 곱 - batch*512 + batch*512 > batch*512*512
                + 단순 과정으로 학습대상 아님
            + attention
                + MultiheadAttention -  batch*512*512 > batch*512*512
                + 학습
            + normalize
                + layerNorm - batch*512*512 > batch*512*512
                + 학습
            + repeat
                + 3차원으로 증가 - batch*3*512*512 > batch*512*512
                + 단순 과정으로 학습대상 아님
            + classifier
                + custom_mobilenet - batch*3*512*512 > batch*5
                + 기존 mobilenet_v3_small 가중치 사용
                + 분류 레이어만 5 class로 줄여 학습 
        + 기존 mobilenet_v3_small 가중치 사용
        + 인코더와 디코더 모두 미리 학습시켜 사용
        + encoder_normalize, attention, normalize, classifier 학습
    
작성자: 윤성진
'''

import torch.nn as nn

# clip - 다운 받아야됨 
import clip

# 자체 라이브러리
from models.encoder.simple_gelu_ae import SimpleGELUEAE
import models.classifier.custom_mobile_net as cmn


class  Kostantiniyye(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, num_heads=8, device='cpu'):
        super(Kostantiniyye, self).__init__()
        clip_encoder, _ = clip.load("ViT-B/32", device=device)
        
        self.image_encoder = clip_encoder.encode_image
        self.portion_encoder = SimpleGELUEAE(
            input_dim=portion_dim,
            latent_dim=latent_dim
            ).encoder
        self.encoder_normalize = nn.LayerNorm(latent_dim)
        
        self.dropout2 = nn.Dropout2d(0.3)
        self.attention = nn.MultiheadAttention(latent_dim, num_heads)
        
        self.normalize = nn.LayerNorm(latent_dim)
        self.dropout3 = nn.Dropout3d(p = 0.3)
        self.classifier = cmn.CustomMobileNet(5)

    def forward(self, vision, portion):
        # 각각 인코딩 후 안정화 - clip은 모델 끝에서 안정화 시키기에 추가로 할 필요 없음
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        portion = self.encoder_normalize(portion)
        
        # 어텐션 적용 후 노멀라이즈
        embed = vision.unsqueeze(2)  * portion.unsqueeze(1) 
        batch_size, h, w = embed.size()
        embed = embed.permute(1, 0, 2)  
        
        attn_output, _ = self.attention(embed, embed, embed)
        x = self.dropout2(attn_output)

        attn_output = attn_output.permute(1, 0, 2)  
        attn_output = attn_output.view(batch_size, h, w)

        x = self.normalize(attn_output)
        
        # 채널 증폭 후 모바일넷    
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.dropout3(x)
        result = self.classifier(x)
        
        return result