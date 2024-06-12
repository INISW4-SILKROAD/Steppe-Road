import torch.nn as nn

# imagebind
import encoder.custom_ibvis_encoder as cibv

class Constantinople(nn.Module):
    def __init__(self, vision_embed_dim = 512, portion_dim = 12, output_dim = 4):
        super(Constantinople, self).__init__()
        self.image_encoder = cibv.CustomIbvisEncoder(out_embed_dim=vision_embed_dim)
        self.polling = nn.Bilinear(vision_embed_dim, portion_dim, output_dim)
        
    def forward(self, x_1, x_2):
        x_1 = self.image_encoder(x_1)
        result = self.polling(x_1, x_2)
        return result
    