
'''
galata.py
촉감 추정 모델
어텐션부터 각 촉감으로 갈라짐
학습 가능하게 변경

class:
    Galata:
        전체 촉감 추정용 모델
        + 각 촉감에 대해 별개의 모델을 적용함
            + 피처맵에 어텐션 메커니즘을 적용
            + 모바일넷에 적용
        + 레이어 정보
            + kostatiniyye와 동일
            + 어텐션부터 나누어져 적용됨

작성자: 윤성진
'''

import torch
import torch.nn as nn


from imagebind.data import load_and_transform_vision_data
from models.encoder.simple_gelu_ae import SimpleGELUEAE
from models.encoder.custom_ibvis_encoder import CustomIbvisEncoder
from models.classifier.attetion_classifier import AttentionClassifier

class Galata(nn.Module):         
    def __init__(self, latent_dim = 512, portion_dim = 12, device='cpu'):
        super(Galata, self).__init__()
        self.device = device
        
        self.preprocessor = load_and_transform_vision_data
        
        self.image_encoder = CustomIbvisEncoder()
        self.portion_encoder = SimpleGELUEAE(
            input_dim=portion_dim,
            latent_dim=latent_dim
            ).encoder
        self.encoder_normalize = nn.LayerNorm(latent_dim)
        
        self.softness = AttentionClassifier()
        self.smoothness = AttentionClassifier()
        self.thickness = AttentionClassifier()
        self.flexibility = AttentionClassifier()
        
        self.touch_classifier = {
            'softness' : self.softness, 
            'smoothness':self.smoothness, 
            'thickness':self.thickness, 
            'flexibility':self.flexibility
            }

    def forward(self, image, portion, touch):
        vision = self.image_encoder(image)
        portion = self.portion_encoder(portion)
        portion = self.encoder_normalize(portion)

        # 행렬 곱
        embed = vision.unsqueeze(2)  * portion.unsqueeze(1) 
        
        touch = self.touch_classifier[touch](embed)
        
        return touch
    
    @torch.no_grad()
    def inference(self, image_path, portion):
        # 각각 인코딩 후 안정화 - clip은 모델 끝에서 안정화 시키기에 추가로 할 필요 없음
        image = self.preprocessor([image_path], self.device)
        vision = self.image_encoder(image)
        
        tenportion = torch.Tensor([portion]).to(self.device)
        portion = self.portion_encoder(tenportion)
        portion = self.encoder_normalize(portion)

        # 행렬 곱
        embed = vision.unsqueeze(2)  * portion.unsqueeze(1) 
        
        softness = self.softness(embed)
        smoothness = self.smoothness(embed)
        thickness = self.thickness(embed)
        flexibility = self.flexibility(embed)
        
        result = (
            softness.max(dim = -1)[1].item(), 
            smoothness.max(dim = -1)[1].item(), 
            thickness.max(dim = -1)[1].item(), 
            flexibility.max(dim = -1)[1].item()
        )
        
        return result
    
    def freeze(self, touch):
        names = [
            f'{touch}.attention.in_proj_weight', 
            f'{touch}.attention.in_proj_bias',
            f'{touch}.attention.out_proj.weight',
            f'{touch}.attention.out_proj.bias',
            f'{touch}.normalize.weight',
            f'{touch}.normalize.bias', 
            f'{touch}.classifier.mobilenet.classifier.0.weight',
            f'{touch}.classifier.mobilenet.classifier.0.bias',
            f'{touch}.classifier.mobilenet.classifier.3.weight',
            f'{touch}.classifier.mobilenet.classifier.3.bias'
            ]        
    
        for name, param in self.named_parameters():
            if name in names:
                param.requires_grad = True
            else:
                param.requires_grad = False
        