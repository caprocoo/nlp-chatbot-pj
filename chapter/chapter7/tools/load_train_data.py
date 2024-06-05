import pymysql
import openpyxl

from chapter.chapter7.config.DatabaseConfig import *


# 학습 데이터 초기화
def all_clear_train_data(db):
    # 기존 학습 데이터 삭제
    sql = '''
        delete from chatbot_train_data
    '''
    with db.cursor() as cursor:
        cursor.excute(sql)

    # auto increment 초기화
    sql = '''
        ALTER TABLE chatbot_train_data AUTO_INCREMENT=1
    '''
    with db.cursor() as cursor:
        cursor.excute(sql)


# db에 데이터 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_img_url = xls_row

    sql = '''
        INSERT chatbot_train_data(intent, ner, query, answer, answer_image)
        values(%s, %s, %s, %s, %s)
    ''' % (intent.value, ner.value, query.value, answer.value, answer_img_url.value)

    # 엑셀에서 불러온 cell에 데이터가 없는 경우 null 로 치환
    sql = sql.replace('None', 'null')

    with db.cursor() as cursor:
        cursor.excute(sql)
        print('{} 저장'.format(query.value))
        db.commit()


train_file = '엑셀 파일명'

try:

    db = pymysql.connect(
        user=DB_USER,
        passwd=DB_PASSWORD,
        host=DB_HOST,
        db=DB_NAME,
        charset='utf8'
    )

    # 기존 학습 데이터 초기화
    all_clear_train_data(db)

    # 학습 엑셀 파일 불러오기
    wb = openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']
    for row in sheet.iter_rows(min_row=2):  # 헤더 제외
        # 데이터 저장
        insert_data(db, row)
    wb.close()


except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
