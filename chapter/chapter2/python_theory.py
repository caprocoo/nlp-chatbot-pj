# Chapter 3. 토크나이징 (p.101 ~ p.)
# KoNLPy (자연어 처리를 위한 파이썬 라이브러리)

# 목차
#   1. Kkma (문장에 따라 정확한 형태소 분석이 안 될 수도 있다.)
#   2. Komoran (문장에 따라 정확한 형태소 분석이 안 될 수도 있다.)


from konlpy.tag import Kkma
from konlpy.tag import Komoran


# # 1. Kkma
#
# # 꼬꼬마 형태소 분석기 객체 생성
# kkma = Kkma()
# text = "아버지가 방에 들어갑니다."
#
# # 형태소 추출 ➊
# # text 변수에 저장된 문장을 형태소 단위로 토크나이징함.
# morphs = kkma.morphs(text)
# print('morphs, ', morphs)
#
# # 형태소와 품사 태그 추출 ➋
# # text 변수에 저장된 문장을 품사 태깅함.
# pos = kkma.pos(text)
# print('pos, ', pos)
#
# # 명사만 추출 ➌
# nouns = kkma.nouns(text)
# print('nouns, ', nouns)
#
# # 문장 분리 ➍
# sentences = "오늘 날씨는 어때요? 내일은 덥다던데."
# s = kkma.sentences(sentences)
# print('sentences, ', s)


# 2. Komoran
# Komoran이 Kkma보다 형태소를 빠르게 분석함

# 코모란 형태소 분석기 객체 생성
komoran = Komoran()
text = "아버지가 방에 들어갑니다."

# 형태소 추출 ➊
morphs = komoran.morphs(text)
print(morphs)

# 형태소와 품사 태그 추출 ➋
pos = komoran.pos(text)
print(pos)

# 명사만 추출 ➌
nouns = komoran.nouns(text)
print(nouns)




# Chapter .  (p. ~ p.)

# 목차
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.