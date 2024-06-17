# 모델

+ 인코더와 분류기를 조합하여 만든 모델들입니다.

----

## 혼용률과 원단 사진을 통한 촉감 예측모델 Galata 

+ kostantiniyye 모델 4개를 각각 학습시킨 뒤 결합시켜 만든 모델입니다.


+ 평가용으로만 사용합니다. 학습은 불가능합니다.

+ 인코딩 값의 행렬 곱 이전의 값은 공유합니다.  

---
## 모델 목록
| 모델 명           | 이미지 인코더    | 혼용률 인코더          | 결합 방법(퓨전 레이어)                                     | classifier         | 
| ----------------- | ---------------- | ---------------------- | --------------------------------------------------------- | -------------------|
| venezia           | clip / imagebind |      x                 | 512 + 12  > 512 bilinear polling                          | movilenet_v3_small | 
| constantinople    | clip / imagebind | simple_ae.encoder      | 512 + 512 > 512 bilinear polling                          | simple_ae.decoder  |
| constantinople_v2 | clip / imagebind | simple_ae.encoder      | 512 + 512 > 512 bilinear polling + midprocess             | simple_ae.decoder  |
| kostantiniyye     | clip / imagebind | simple_gelu_ae.encoder | 512 * 512 > 512*512 matrix multiply + attention + dropout | mobilenet_v3_small |
| galata            | iamgebind        | simple_gelu_ae.encoder | 512 * 512 > 512*512 matrix multiply + attention + dropout | mobilenet          |
---

## 참고 사항

+ 가중치

  + galata model의 가중치는 [여기](https://drive.google.com/file/d/1hT9mEhn-OK1lPgtlu3R8clwsCvC-zgav/view?usp=sharing)에서 얻을 수 있습니다.
  
    + 기타 다른 모델의 가중치는 [여기](https://drive.google.com/drive/folders/1CWV27MCpXmerGAHXKg9EHfnTT3vVz1O5?usp=sharing) 있습니다.

    + `load_state_dict`의 `strict=False` 옵션을 사용하면, 다른 모델의 가중치도 쉽게 불일 수 있습니다.
  
  + clip, imagebind의 이미지 인코더와 mobilenet의 가중치는 기존 모델의 것을 그대로 사용했습니다.
  
    + imagebind의 경우, 마지막 레이어를 512개로 줄인 것을 사용합니다

    + mobilenet의 경우, 마지막 classifier의 4번째 레이어만 학습시킵니다
   
+ 인코더

  + [clip](https://github.com/openai/CLIP)은 공개되어있는 "ViT-B/32"모델을 사용합니다. 

  + clip의 인코더는 clip모델의 encode_image를 바인딩하여 사용합니다. 

  + [imagebind](https://github.com/facebookresearch/ImageBind?tab=readme-ov-file)는 기존 모델의 통합 인코더에서 비전 인코더를 분리하여 최종 임베딩 차원을 512차원으로 줄인것을 사용합니다.

  + 혼용률 인코더는 직접 모은 40000개의 혼용률 정보 중 중복을 제외한 2700여개의 데이터를 오토인코더를 통해 학습시킨 뒤, 인코더만 떼어 사용합니다.
 
 + 기타
   + kostantiniyye 중 인코더 명이 안붙은 모델은 드롭아웃이 없는 clip 기반 모델입니다.  
