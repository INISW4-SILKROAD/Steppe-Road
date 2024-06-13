import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import Tensor

from simple_ae import SimpleAE

class SimpleVAE(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim= 1024, latent_dim=512):
        super(SimpleVAE, self).__init__()
        # 인코더 정의
        self.encoder = SimpleAE(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim
            ).encoder
        
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        
        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu:Tensor, logvar:Tensor)->Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# 손실 함수 정의
class VAELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'sum') -> None:
        super(VAELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, target_mean, logvar, beta=1) -> Tensor:
        BCE = F.binary_cross_entropy(input, target, reduction=self.reduction)
        KLD = -0.5 * torch.sum(1 + logvar - target_mean.pow(2) - logvar.exp())
        return BCE + beta*KLD