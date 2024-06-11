# 분류기 전처리기
+ 임의 차원의 벡터를 분류기에 맞게 변환하여 주는 전처리기 모음입니다.
---
# 목록
+ `custon_preprosessor.py` - 임의 길이의 벡터를 3*32*32로 변환해줍니다.
  + fc > 3*3 > relu > batch nomalize
  + 사용되는 분류기
    + `custom_mobilenet`
  
