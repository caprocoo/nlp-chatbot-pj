# Chapter 8. 챗봇 학습툴 만들기 (p.237 ~ p.)

# 목차
#   1. 전처리 과정
#   2. 전처리 테스트
#   3. 
#   4. 
#   5. 
#   6.

# 2. 전처리 테스트

from chapter.chapter8._1_data_preprocessing import Preprocess

sent = "내일 오전 10시에 탕수육 주문하고 싶어"

# 전처리 객체 생성
p = Preprocess(userdic='../../utils/user_dic.tsv')

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태그와 같이 키워드 출력
ret = p.get_keywords(pos, without_tag=False)
print(ret)

# 품사 태그 없이 키워드 출력
ret = p.get_keywords(pos, without_tag=True)
print(ret)