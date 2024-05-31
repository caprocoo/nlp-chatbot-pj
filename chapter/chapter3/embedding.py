# Chapter 3. 임베딩  (p.115 ~ p.)
# 임베딩은 단어나 문장을 수치화해 벡터 공간으로 표현하는 과정을 의미함.

# 목차
#   1. 원-핫 인코딩 (희소표현)
#   2. 분산 표현 (밀집표현)
#   3. Word2Vec (분산 표현 형태의 대표적 모델)

from konlpy.tag import Komoran
import numpy as np

# 1. 원-핫 인코딩 (one-hot encoding)
# 표현하고자 하는 단어의 인덱스 요소만 1이고 나머지 요소는 모두 0으로 표현되는 희소 벡터(or 희소 행렬)이다.
# 희소 표현은 각각의 차원이 독립적인 정보를 지니고 있어 사람이 이해하기에는 직관적인 장점이 있지만
# 단어 사전의 크기가 커질수록 메모리 낭비와 계산 복잡도가 커지는 단점도 있다.
# 또한 단어 간의 연관성이 전혀 없어 의미를 담을 수 없다. (단어 순서별)

komoran = Komoran()
text = "오늘 날씨는 구름이 많아요."

# 명사만 추출 ➊
nouns = komoran.nouns(text)
print(nouns)

# 단어 사전 구축 및 단어별 인덱스 부여 ➋
dics = {}
for word in nouns:
    if word not in dics.keys():
        dics[word] = len(dics)
print(dics)

# 원-핫 인코딩
nb_classes = len(dics)  # 원-핫 벡터 차원의 크기를 결정 ➌
targets = list(dics.values())  # 딕셔너리 -> 리스트 ➍
one_hot_targets = np.eye(nb_classes)[targets]  # eye()는 단위행렬을 만들어 줌 & [target] 을 이용하여 단어 사전의 순서에 맞게 정렬해줌 ➎
print(one_hot_targets)


# 2. 분산 표현 (or 밀집 표현 : 가장 많이 사용하는 방식)
# 희소 표현과 달리 하나의 차원에 다양한 정보를 가지고 있다.
# 1) 임베딩 벡터의 차원을 데이터 손실을 최소화하면서 압축할 수 있다. (차원의 저주 방지)
# 2) 임베딩 벡터에는 단어의 의미, 주변 단어 간의 관계 등 많은 정보가 내포되어 있어 일반화 능력이 뛰어나다.


# 3. Word2Vec (분산 표현 형태의 대표적 모델)
# 1) CBOW 모델
#   : 주변 단어들을 이용해 타깃 단어를 예측하는 신경망 모델
# 2) skip-gram 모델
#   : 하나의 타깃 단어를 이용해 주변 단어들을 예측하는 신경망 모델




# Chapter .  (p. ~ p.)

# 목차
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.
