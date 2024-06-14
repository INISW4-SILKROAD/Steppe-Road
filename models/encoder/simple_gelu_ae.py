import torch.nn as nn

class SimpleGELUEAE(nn.Module):
    def __init__(self, input_dim = 12, hidden_dim = 1024, latent_dim = 512):
        super(SimpleGELUEAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x