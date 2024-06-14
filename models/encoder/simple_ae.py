'''
simple_ae.py
간단한 오토 인코더

class:
    SimpleAE: 
        간단히 구현한 오토 인코더.
        GELU와 batch.norm으로 모델 안정화
        혼용률 임베딩을 위해 사용
        
작성자: 윤성진
'''

import torch.nn as nn

class SimpleAE(nn.Module):
    '''
    SimpleAE: 
    간단히 구현한 오토 인코더. GELU와 batch.norm으로 모델 안정화
    혼용률 임베딩을 위해 사용. 
    저차원에서 고차원으로 확대시킨 뒤, 다시 저차원으로 변환
    
    method:
        encode: 벡터를 잠재공간으로 인코딩함
        decode: 잠재 공간 내의 벡터를 혼용률로 디코딩 함 
    '''
    def __init__(self, input_dim = 12, hidden_dim = 1024, latent_dim = 512):
        super(SimpleAE, self).__init__()
      
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )
          
        self.decoder = nn.Sequential(
              nn.Linear(latent_dim, hidden_dim),
              nn.BatchNorm1d(hidden_dim),
              nn.GELU(),
              nn.Linear(hidden_dim, input_dim),
              nn.BatchNorm1d(input_dim),
              nn.ReLU()
          )
        
    def forward(self, x):
      out = x.view(x.size(0), -1)
      out = self.encoder(out)
      out = self.decoder(out)
      out = out.view(x.size())
      return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)