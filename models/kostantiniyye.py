import torch.nn as nn
import clip
# imagebind
import models.encoder.simple_ae as cae
import models.classifier.custom_mobile_net as cmn


class  Kostantiniyye(nn.Module):
    def __init__(self, latent_dim = 512, portion_dim = 12, touch_dim = 4, num_heads=8, device='cpu'):
        super(Kostantiniyye, self).__init__()
        self.image_encoder, self.preprocessor = clip.load("ViT-B/32", device=device)
        
        self.portion_encoder = cae.SimpleAE(
            input_dim=portion_dim,
            latent_dim=latent_dim
            ).encoder
        self.encoder_normalize = nn.LayerNorm(latent_dim)
        
        self.attention = nn.MultiheadAttention(latent_dim, num_heads)
        self.normalize = nn.LayerNorm(latent_dim)
        
        self.classifier = cmn.CustomMobileNet(5)

    def forward(self, vision, portion):

        vision = self.image_encoder.encode_image(vision)
        portion = self.portion_encoder(portion)
        portion = self.encoder_normalize(portion)
        
        embed = vision.unsqueeze(2)  * portion.unsqueeze(1) 
        batch_size, h, w = embed.size()
        embed = embed.permute(1, 0, 2)  
        
        attn_output, _ = self.attention(embed, embed, embed)
        
        attn_output = attn_output.permute(1, 0, 2)  
        attn_output = attn_output.view(batch_size, h, w)
        
        x = self.normalize(attn_output)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)

        result = self.classifier(x)
        return result