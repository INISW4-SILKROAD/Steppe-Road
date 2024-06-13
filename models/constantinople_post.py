import torch.nn as nn

# imagebind
import models.encoder.simple_ae as cae

class ConstantinoplePost(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12):
        super(ConstantinoplePost, self).__init__()
        self.portion_encoder = cae.SimpleAE(
            input_dim=portion_dim, 
            latent_dim=latent_dim
            ).encoder
        
        self.polling = nn.Bilinear(
            in1_features=latent_dim, 
            in2_features=latent_dim, 
            out_features=latent_dim
            )
        
    def forward(self, embeded_vision, portion):
        portion = self.portion_encoder(portion)
        latent = self.polling(embeded_vision, portion)
        return latent