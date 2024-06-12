import torch.nn as nn

# imagebind
import encoder.custom_ibvis_encoder as cibv
import encoder.custom_autoencoder as cae

class Constantinople(nn.Module):
    def __init__(self, vision_embed_dim = 512, portion_dim = 12, output_dim = 4):
        super(Constantinople, self).__init__()
        self.image_encoder = cibv.CustomIbvisEncoder(out_embed_dim=vision_embed_dim)
        self.portion_encoder = cae.CustomAutoEncoder().encoder
        self.polling = nn.Bilinear(vision_embed_dim, vision_embed_dim, vision_embed_dim)
        self.touch_decoder = cae.CustomAutoEncoder(4).decoder
        
        
    def forward(self, vision, portion):
        vision = self.image_encoder(vision)
        portion = self.portion_encoder(portion)
        latent = self.polling(vision, portion)
        result = self.touch_decoder(latent)
        return result
    