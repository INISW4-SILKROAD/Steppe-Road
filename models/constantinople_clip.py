import torch.nn as nn

# imagebind
import clip
import models.encoder.simple_ae as cae

class ConstantinopleClip(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, touch_dim = 4):
        super(ConstantinopleClip, self).__init__()
        clip_model, preprocesser = clip('ViT-B/32', device)
        self.preprocesser = preprocesser
        self.image_encoder = clip_model
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
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        
        latent = self.polling(vision, portion)
        latent = self.postprocess(latent)
        
        result = self.touch_decoder(latent)
        return result