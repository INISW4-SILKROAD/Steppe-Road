import torch
import torch.nn as nn
import os

# imagebind

import encoder.custom_ibvis_encoder as cibv
from classifier.custom_mobile_net import CustomMobileNet

class Venezia(nn.Module):
    def __init__(self, vision_embed_dim = 512, portion_dim = 12, output_dim = 512):
        super(Venezia, self).__init__()
        self.image_encoder = cibv.CustomIbvisEncoder(out_embed_dim=vision_embed_dim)
        self.polling = nn.Bilinear(vision_embed_dim, portion_dim, output_dim)
        self.classifier_ = CustomMobileNet()
        
        
    def forward(self, x_1, x_2):
        x_1 = self.image_encoder(x_1)
        result = self.polling(x_1, x_2)
        result = self.classifier_(result)
        return result

def load_venezia_pretrain(out_embed_dim = 512):
    model = Venezia()
    weight_path = f".checkpoints/pretrained_venezia_{out_embed_dim}.pth"
    if not os.path.exists(weight_path):
        print('WARNING: no checkpoint exist - cant load weight')
        return None

    model.load_state_dict(torch.load(weight_path), strict=False)
    return model