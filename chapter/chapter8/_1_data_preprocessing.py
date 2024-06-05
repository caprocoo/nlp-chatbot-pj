# Chapter 8. 챗봇 학습툴 만들기 (p.237 ~ p.)

# 목차
#   1. 전처리 과정
#   2. 전처리 테스트
#   3. 
#   4. 
#   5. 
#   6.

# 1. 전처리 과정

from konlpy.tag import Komoran


class Preprocess:
    def __init__(self, userdic=None):  # 생성자
        # 형태소 분석기 초기화 ➊
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사 ➋
        # 참조 : https://docs.komoran.kr/firststep/postypes.html
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

        # 형태소 분석기 POS 태거 ➌
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    # 불용어 제거 후 필요한 품사 정보만 가져오기 ➍
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list
