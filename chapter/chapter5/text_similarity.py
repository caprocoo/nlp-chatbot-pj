# Chapter 5. 텍스트 유사도 (p.129 ~ p.140)
# 두 문장 간의 유사도를 계산하기 위해서는 문장 내에 존재하는 단어들을 수치화해야 한다.
# 언어 모델에 따라 통계를 이용하는 방법(n-gram)과 인공 신경망을 이용하는 방법(WordVec)으로 나눌 수 있다.

# 목차
#   1. n-gram 유사도
#   2. 코사인 유사도


# 1. n-gram 유사도
#  : 이웃한 단어의 출현 횟수를 통계적으로 표현해 텍스트의 유사도를 계산
#  : 문장에서 n개의 단어를 토큰으로 사용

# from konlpy.tag import Komoran
#
# # 어절 단위 n-gram ➊
# # 추출된 토큰들은 튜플 형태로 반환된다.
# def word_ngram(bow, num_gram):
#  text = tuple(bow)
#  ngrams = [text[x:x + num_gram] for x in range(0, len(text))]
#  return tuple(ngrams)
#
# # 유사도 계산 ➋
# def similarity(doc1, doc2):
#  cnt = 0
#  for token in doc1:
#      if token in doc2:
#          cnt = cnt + 1
#  return cnt/len(doc1)
#
# # 문장 정의
# sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다.'
# sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학했다.'
# sentence3 = '나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다.'
#
# # 형태소 분석기에서 명사(단어) 추출 ➌
# komoran = Komoran()
# bow1 = komoran.nouns(sentence1)
# bow2 = komoran.nouns(sentence2)
# bow3 = komoran.nouns(sentence3)
#
# # 단어 n- gram 토큰 추출 ➍
# doc1 = word_ngram(bow1, 2) # 2- gram 방식으로 추출
# doc2 = word_ngram(bow2, 2)
# doc3 = word_ngram(bow3, 2)
#
# # 추출된 n- gram 토큰 출력
# print(doc1)
# print(doc2)
#
# # 유사도 계산
# r1 = similarity(doc1, doc2) # ➎
# r2 = similarity(doc3, doc1) # ➏
#
# # 계산된 유사도 출력
# print(r1)
# print(r2)
#
# # n-gram 은 문장에 존재하는 모든 단어의 출현 빈도를 확인하는 것이 아니라
# # 연속되는 문장에서 일부 단어(n)만 확인하다 보니 전체 문장을 고려한 언어 모델보다 정확도가 떨어질 수 있다.
#
# # n을 작게 잡을수록 카운트 확률은 높아지지만 문맥을 파악하는 정확도는 떨어질 수 밖에 없다.
# # 그래서 n-gram 에서는 n을 보통 1~5 사이의 값을 많이 사용한다.



# 2. 코사인 유사도
# 두 벡터 간 코사인 각도를 이용해 유사도를 측정하는 방법이다.
# 일반적으로 벡터의 크기가 중요하지 않을 때 그 거리를 측정하기 위해 사용한다.

# 코사인은 -1~1 사이의 값을 가지며, 두 벡터의 방향이 완전히 동일한 경우에는 1,
# 반대 방향인 경우에는 -1, 두 벡터가 서로 직각을 이루면 0의 값을 가진다.
# 즉, 두 벡터의 방향이 같아질수록 유사하다 볼 수 있다.

from konlpy.tag import Komoran
import numpy as np  
from numpy import dot
from numpy.linalg import norm

# 코사인 유사도 계산
def cos_sim(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# TDM 만들기
def make_term_doc_mat(sentence_bow, word_dics):
    freq_mat = {}
    
    for word in word_dics: 
        freq_mat[word] = 0
    
    for word in word_dics:    
        if word in sentence_bow:
            freq_mat[word] +=1
            
    return freq_mat

# 단어 벡터 만들기    
def make_vector(tdm):
    vec=[]
    for key in tdm:
        vec.append(tdm[key])
    return vec

# 문장 정의
sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다.'
sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학했다.'
sentence3 = '나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다.'

# 형태소 분석기를 이용해 단어 묶음 리스트 생성
komoran = Komoran()
bow1 = komoran.nouns(sentence1)
bow2 = komoran.nouns(sentence2)
bow3 = komoran.nouns(sentence3)

# 단어 묶음 리스트를 하나로 합침
bow = bow1 + bow2 + bow3

# 단어 묶음에서 중복을 제거해 단어 사전 구축
word_dics = []
for token in bow:
    if token not in word_dics:
        word_dics.append(token)


# 문장별 단어 문서 행렬 계산
freq_list1 = make_term_doc_mat(bow1, word_dics)
freq_list2 = make_term_doc_mat(bow2, word_dics)
freq_list3 = make_term_doc_mat(bow3, word_dics)
print(freq_list1)
print(freq_list2)
print(freq_list3)

# 문장 벡터 생성
doc1 = np.array(make_vector(freq_list1))
doc2 = np.array(make_vector(freq_list2))
doc3 = np.array(make_vector(freq_list3))
print(doc1)
print(doc2)
print(doc3)

# 코사인 유사도 계산
r1 = cos_sim(doc1, doc2)
r2 = cos_sim(doc3, doc1)
print(r1)
print(r2)


# 챗봇 엔진에 어떤 질문이 입력되었을 때 우리는 해당 질문에 적절한 답변을 출력할 수 있어야 한다.
# 이때 입력된 질문과 시스템에 저장되어 있는 질문-답변 데이터의 유사도를 계산할 수 있어야 해당 질문에 연관된 답변을 내보낼 수 있다.

# Chapter .  (p. ~ p.)

# 목차
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.