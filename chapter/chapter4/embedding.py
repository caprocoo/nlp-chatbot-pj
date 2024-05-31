# Chapter 3. 임베딩  (p.115 ~ p.128)
# 임베딩은 단어나 문장을 수치화해 벡터 공간으로 표현하는 과정을 의미함.

# 목차
#   1. 원-핫 인코딩 (희소표현)
#   2. 분산 표현 (밀집표현)
#   3. Word2Vec (분산 표현 형태의 대표적 모델)

from konlpy.tag import Komoran
import numpy as np

from gensim.models import Word2Vec
import time

# 1. 원-핫 인코딩 (one-hot encoding)
# 표현하고자 하는 단어의 인덱스 요소만 1이고 나머지 요소는 모두 0으로 표현되는 희소 벡터(or 희소 행렬)이다.
# 희소 표현은 각각의 차원이 독립적인 정보를 지니고 있어 사람이 이해하기에는 직관적인 장점이 있지만
# 단어 사전의 크기가 커질수록 메모리 낭비와 계산 복잡도가 커지는 단점도 있다.
# 또한 단어 간의 연관성이 전혀 없어 의미를 담을 수 없다. (단어 순서별)

# komoran = Komoran()
# text = "오늘 날씨는 구름이 많아요."
#
# # 명사만 추출 ➊
# nouns = komoran.nouns(text)
# print(nouns)
#
# # 단어 사전 구축 및 단어별 인덱스 부여 ➋
# dics = {}
# for word in nouns:
#     if word not in dics.keys():
#         dics[word] = len(dics)
# print(dics)
#
# # 원-핫 인코딩
# nb_classes = len(dics)  # 원-핫 벡터 차원의 크기를 결정 ➌
# targets = list(dics.values())  # 딕셔너리 -> 리스트 ➍
# one_hot_targets = np.eye(nb_classes)[targets]  # eye()는 단위행렬을 만들어 줌 & [target] 을 이용하여 단어 사전의 순서에 맞게 정렬해줌 ➎
# print(one_hot_targets)


# 2. 분산 표현 (or 밀집 표현 : 가장 많이 사용하는 방식)
# 희소 표현과 달리 하나의 차원에 다양한 정보를 가지고 있다.
# 1) 임베딩 벡터의 차원을 데이터 손실을 최소화하면서 압축할 수 있다. (차원의 저주 방지)
# 2) 임베딩 벡터에는 단어의 의미, 주변 단어 간의 관계 등 많은 정보가 내포되어 있어 일반화 능력이 뛰어나다.


# 3. Word2Vec (분산 표현 형태의 대표적 모델)
# 1) CBOW 모델
#   : 주변 단어들을 이용해 타깃 단어를 예측하는 신경망 모델
# 2) skip-gram 모델
#   : 하나의 타깃 단어를 이용해 주변 단어들을 예측하는 신경망 모델


# 3.1. Word2Vec 모델 학습

# # 네이버 영화 리뷰 데이터 읽어옴
# def read_review_data(filename):
#     with open(filename, 'r', encoding='UTF8') as f:
#         data = [line.split('\t') for line in f.read().splitlines()]
#         data = data[1:] # 헤더 제거
#     return data
#
# # 학습 시간 측정 시작
# start = time.time()
#
# # 리뷰 파일 읽어오기
# print('1) 말뭉치 데이터 읽기 시작')
# review_data = read_review_data('../../ratings.txt') # https://github.com/e9t/nsmc 참조
# print(len(review_data))
# print('1) 말뭉치 데이터 읽기 완료 : ', time.time() - start)
#
# # 문장 단위로 명사만 추출해 학습 입력 데이터로 만듦
# print('2) 형태소에서 명사만 추출 시작')
# komoran = Komoran()
# docs = [komoran.nouns(sentence[1]) for sentence in review_data]
#
#
# print('2) 형태소에서 명사만 추출 완료 : ', time.time() - start)
#
# # Word2Vec 모델 학습
# print('3) Word2Vec 모델 학습 시작')
# model = Word2Vec(sentences=docs, vector_size=200, window=4, hs=1, min_count=2, sg=1)
# print('3) Word2Vec 모델 학습 완료 : ', time.time() - start)
#
# # 모델 저장
# print('4) 학습된 모델 저장 시작')
# model.save('nvmc.model')
# print('4) 학습된 모델 저장 완료 : ', time.time() - start)
#
# #학습된 말뭉치 수, 코퍼스 내 전체 단어 수
# print('corpus_count : ', model.corpus_count)
# print('corpus_total_words : ', model.corpus_total_words)



# 3.2. Word2Vec 모델 활용

# 모델 로딩
model = Word2Vec.load('nvmc.model')
print('corpus_total_words : ', model.corpus_total_words)

# '사랑'이란 단어로 생성한 단어 임베딩 벡터
print('사랑 : ', model.wv['사랑'])

# 단어 유사도 계산
print('일요일 = 월요일\t', model.wv.similarity(w1='일요일', w2='월요일'))
print('안성기 = 배우\t', model.wv.similarity(w1='안성기', w2='배우'))
print('대기업 = 삼성\t', model.wv.similarity(w1='대기업', w2='삼성'))
print('일요일 != 삼성\t', model.wv.similarity(w1='일요일', w2='삼성'))
print('히어로 != 삼성\t', model.wv.similarity(w1='히어로', w2='삼성'))

# 가장 유사한 단어 추출
print(model.wv.most_similar('안성기', topn=5))
print(model.wv.most_similar('시리즈', topn=5))

# 놀라울 정도로 유사한 단어를 찾는 경우도 있지만 이해하기 힘든 결과를 출력하는 경우도 있다.
# 이는 주제에 맞는 말뭉치 데이터가 부족해서 생기는 현상이니 품질 좋은 말뭉치 데이터를 학습하면 임베딩 성능이 많이 좋아진다.





# Chapter .  (p. ~ p.)

# 목차
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.
