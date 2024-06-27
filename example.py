import torch
from models.galata import Galata

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Galata(device = device)
model.load_state_dict(torch.load('galata.pth'))
model.to(device)
model.eval()

# 이미지는 경로를 입력해주세요 
# 혼용률은 12개의 피처를 가지는 1차원 리스트입니다. 
result = model('example/90.jpg', [ 0.0, 0.77, 0.0, 0.0, 0.18, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print(result) #(3, 2, 2, 0)