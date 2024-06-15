'''
istanbul.py
촉감 추정 모델(작동용)
미리 학습시킨 4개 모델 통합버전
어텐션부터 각 촉감으로 갈라짐
학습 불가능함. 명심할 것 - 배포용으로 만드는 파일.

class:
    Istanbul:
        전체 촉감 추정용 모델
        + 각 촉감에 대해 별개의 모델을 적용함
            + 피처맵에 어텐션 메커니즘을 적용
            + 모바일넷에 적용
        + 레이어 정보
            + kostatiniyye와 동일
            + 어텐션부터 각 촉감을 지원

작성자: 윤성진
'''

import torch
import torch.nn as nn
from PIL import Image

import clip

from imagebind.data import load_and_transform_vision_data
from models.encoder.simple_gelu_ae import SimpleGELUEAE

from models.classifier.attetion_classifier import AttentionClassifier

class Galata(nn.Module):

            
    def __init__(self, latent_dim = 512, portion_dim = 12, device='cpu'):
        super(Galata, self).__init__()
        self.device = device
        self.preprocessor = load_and_transform_vision_data
        
        clip, _= clip.load("ViT-B/32", device=device)
        self.image_encder = clip.encode_image
        self.portion_encoder = SimpleGELUEAE(
            input_dim=portion_dim,
            latent_dim=latent_dim
            ).encoder
        self.encoder_normalize = nn.LayerNorm(latent_dim)
        
        self.softness = AttentionClassifier()
        self.smoothness = AttentionClassifier()
        self.thickness = AttentionClassifier()
        self.flexibility = AttentionClassifier()

    @torch.no_grad()
    def forward(self, image_path, portion):
        # 각각 인코딩 후 안정화 - clip은 모델 끝에서 안정화 시키기에 추가로 할 필요 없음
        image = self.preprocessor([image_path], self.device)
        vision = self.image_encoder(image)
        portion = self.portion_encoder(portion)
        portion = self.encoder_normalize(portion)

        # 행렬 곱
        embed = vision.unsqueeze(2)  * portion.unsqueeze(1) 
        
        softness = self.softness(embed).max(dim=1)[1].tolist()[0]
        smoothness = self.smoothness(embed).max(dim=1)[1].tolist()[0]
        thickness = self.thickness(embed).max(dim=1)[1].tolist()[0]
        flexibility = self.flexibility(embed).max(dim=1)[1].tolist()[0]
        
        return softness, smoothness, thickness, flexibility