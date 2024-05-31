# Chapter 7. 챗봇 학습툴 만들기 (p.215 ~ p.225)

# 목차
#   1. db 호스트 연결 및 닫기 (mysql)
#   2. 테이블 생성 sql 정의 ➊
#   3. 데이터 삽입
#   4. 데이터 변경
#   5. 데이터 삭제
#   6. 데이터 조회


import pymysql
import pandas as pd


chatbot_db = None

try:
    # 1. db 호스트 연결 및 닫기
    chatbot_db = pymysql.connect(
        user='meta',
        passwd='meta1234!@#$',
        host='127.0.0.1',
        db='chatbot',
        charset='utf8'
    )
    print('db connected')

    # 2. 테이블 생성 sql 정의 ➊
    # sql = '''
    #  CREATE TABLE tb_student (
    #  id int primary key auto_increment not null,
    #  name varchar(32),
    #  age int,
    #  address varchar(32)
    #  ) ENGINE=InnoDB DEFAULT CHARSET=utf8
    #  '''
    # # 테이블 생성 ➋
    # with chatbot_db.cursor() as cursor:
    #     cursor.execute(sql)
    
    # 3. 데이터 삽입
    # sql = '''
    # INSERT tb_student(name, age, address) values('Kei', 35, 'Korea')
    # '''
    # # 데이터 삽입
    # with chatbot_db.cursor() as cursor:
    #     cursor.execute(sql)
    # chatbot_db.commit()

    # 4. 데이터 변경
    # # 데이터 수정 sql 정의 ➊
    # id = 1  # 데이터 id(Primary Key)
    # sql = '''
    #  UPDATE tb_student set name="케이", age=36 where id=%d
    #  ''' % id
    # # 데이터 수정 ➋
    # with chatbot_db.cursor() as cursor:
    #     cursor.execute(sql)
    #     chatbot_db.commit()

    # 5. 데이터 삭제
    # # 데이터 삭제 sql 정의 ➊
    # id = 1  # 데이터 id(Primary Key)
    # sql = '''
    #  DELETE from tb_student where id=%d
    #  ''' % id
    # # 데이터 삭제 ➋
    # with chatbot_db.cursor() as cursor:
    #     cursor.execute(sql)
    # chatbot_db.commit()

    # 6. 데이터 조회
    # 데이터 db에 추가 ➊
    students = [
        {'name': 'Kei', 'age': 36, 'address': 'PUSAN'},
        {'name': 'Tony', 'age': 34, 'address': 'PUSAN'},
        {'name': 'Jaeyoo', 'age': 39, 'address': 'GWANGJU'},
        {'name': 'Grace', 'age': 28, 'address': 'SEOUL'},
        {'name': 'Jenny', 'age': 27, 'address': 'SEOUL'},
    ]
    for s in students:
        with chatbot_db.cursor() as cursor:
            sql = '''
                    insert tb_student(name, age, address) values("%s",%d,"%s")
                    ''' % (s['name'], s['age'], s['address'])
            cursor.execute(sql)
        chatbot_db.commit() # 커밋

    # 30대 학생만 조회 ➋
    cond_age = 30
    with chatbot_db.cursor(pymysql.cursors.DictCursor) as cursor:
        sql = '''
         select * from tb_student where age > %d
         ''' % cond_age
        cursor.execute(sql)
        results = cursor.fetchall()
    print(results)

    # 이름 검색 ➌
    cond_name = 'Grace'
    with chatbot_db.cursor(pymysql.cursors.DictCursor) as cursor:
        sql = '''
         select * from tb_student where name="%s"
         ''' % cond_name
        cursor.execute(sql)
        result = cursor.fetchone()
    print(result['name'], result['age'])

    # pandas 데이터프레임으로 표현 ➍
    df = pd.DataFrame(results)
    print(df)


     

except Exception as e:
    print(e)

finally:
     if chatbot_db is not None:
        chatbot_db.close()

