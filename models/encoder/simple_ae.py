import torch.nn as nn

class SimpleAE(nn.Module):
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
  
  def get_codes(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)