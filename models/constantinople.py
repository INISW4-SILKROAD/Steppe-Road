import torch.nn as nn

# imagebind
import encoder.custom_ibvis_encoder as cibv
import models.encoder.simple_ae as cae

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
        
        self.postprocess = nn.Sequential(
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
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        
        latent = self.polling(vision, portion)
        latent = self.postprocess(latent)
        
        result = self.touch_decoder(latent)
        return result