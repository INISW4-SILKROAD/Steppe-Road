# Steppe-Road
+ 촉감 추정 모델을 만들기위한 서브 모델을 모아두는 레포지토리 입니다. 
  + utils - 기타 편의 기능을 제공하는 코드를 모아둡니다. 
  + models - 조합한 촉감 분류 모델을 모아둡니다. 
    + encoder - 직물과 혼용률을 임베딩 하는 인코더를 모아둡니다.
    + classifier - 분류기를 모아둡니다.
---  
# 촉감 예측 모델 사용 방법
## 환경 구성 
1. GPU - CUDA 11.6
2. 가상환경 생성 
  ```
  $ conda create -n steppe-road python=3.8 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

  ```
3. 참고 코드 클론
  + meta - [ImageBind](https://github.com/facebookresearch/ImageBind?tab=readme-ov-file)
  + openai - [Clip](https://github.com/openai/CLIP)
  ```terminal
  $ git clone https://github.com/facebookresearch/ImageBind.git
  $ cd ImageBind
  $ pip install .
  $ cd ..
  $ pip install ftfy regex tqdm
  $ pip install git+https://github.com/openai/CLIP.git
  ```
  + 설치 안되는 경우, 아나콘다에서 cartropy 받기

4. git 레포 복사
  ```terminal
  $ git clone https://github.com/INISW4-SILKROAD/Steppe-Road.git  
  cd Steppe-Road
  ```
# 실행 방법
+ Steppe-Road 폴더 안의 py또는 ipynb파일을 통해 실행해야 합니다. 
+ 다음 링크에서 가중치([galata.pth](https://drive.google.com/file/d/1hT9mEhn-OK1lPgtlu3R8clwsCvC-zgav/view?usp=sharing))를 다운받아 Steppe-Road 안에 넣어주세요
+ 다음과 같이 실행해주세요
```python
import torch
from models.galata import Galata

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Galata(device = device)
model.load_state_dict(torch.load('galata.pth'))

model.eval()

# 이미지는 경로를 입력해주세요 
# 혼용률은 12개의 피처를 가지는 1차원 리스트입니다. 
result = model('example/90.jpg', [ 0.0, 0.77, 0.0, 0.0, 0.18, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print(result) #(3, 2, 2, 0)
```
+ 또는 example.ipynb를 통해 실행할 수도 있습니다. 
