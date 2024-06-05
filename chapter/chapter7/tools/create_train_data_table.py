import pymysql
from chapter.chapter7.config.DatabaseConfig import *


chatbot_db = None

try:

    chatbot_db = pymysql.connect(
        user=DB_USER,
        passwd=DB_PASSWORD,
        host=DB_HOST,
        db=DB_NAME,
        charset='utf8'
    )
    print('db connected')


    # intent : 질문의 의도를 나타내는 텍스트. 의도가 없는 경우 비워둔다.
    # ner : 질문에 필요한 개체명. 개체명이 없는 경우 비워둔다.
    # query : 질문 텍스트
    # answer : 답변 텍스트
    # answer_image : 답변에 들어갈 이미지 URL. 이미지 URL이 없는 경우 비워둔다.
    sql = '''
     CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
     `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
     `intent` varchar(45) NULL,
     `ner` varchar(1024) NULL,
     `query` TEXT NULL,
     `answer` TEXT NOT NULL,
     `answer_image` varchar(2048) NULL,
     PRIMARY KEY (`id`)) 
     ENGINE=InnoDB DEFAULT CHARSET=utf8
     '''

    # 테이블 생성 ➋
    with chatbot_db.cursor() as cursor:
        cursor.execute(sql)


except Exception as e:
    print(e)

finally:
     if chatbot_db is not None:
        chatbot_db.close()