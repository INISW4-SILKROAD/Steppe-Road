"""
attetion_classifier.py
kostantiniyye 학습시 촉감마다 따로 학습시키는 부분

class:
    AttetionClassifier: 
        + 레이어 구조
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
         
작성자: 윤성진
"""
import torch
import torch.nn as nn

from models.classifier.custom_mobile_net import CustomMobileNet

class AttentionClassifier(nn.Module):
    def __init__(self, num_class = 5, latent_dim = 512, num_heads=8):
        super(AttentionClassifier, self).__init__()

        self.attention = nn.MultiheadAttention(latent_dim, num_heads)
        self.normalize = nn.LayerNorm(latent_dim)
        self.classifier = CustomMobileNet(num_class)

    @torch.no_grad()
    def forward(self, embed):
        # 어텐션 적용 후 노멀라이즈
        batch_size, h, w = embed.size()
        embed = embed.permute(1, 0, 2)  
        
        attn_output, _ = self.attention(embed, embed, embed)
        
        attn_output = attn_output.permute(1, 0, 2)  
        attn_output = attn_output.view(batch_size, h, w)
    
        x = self.normalize(attn_output)
        
        # 채널 증폭 후 모바일넷    
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        result = self.classifier(x)
        
        return result