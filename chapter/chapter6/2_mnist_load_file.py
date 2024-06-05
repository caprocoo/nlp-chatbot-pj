# Chapter 6. 챗봇 엔진에 필요한 딥러닝 모델

# 목차
#   1. MNIST 분류모델 학습
#   2. 학습된 MNIST 분류 모델 파일 불러와 사용하기
#   3. 문장 분류를 위한 CNN 모델
#   4. 학습된 CNN 분류 모델 파일 불러와 사용하기
#   5. RNN 모델
#   6. LSTM 모델
#   7. 양방향 LSTM 모델
#   8. 개체명 인식 (NER)


# 2. 학습된 MNIST 분류 모델 파일 불러오기

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# MNIST 데이터셋 가져오기 ➊
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0 # 데이터 정규화
# 모델 불러오기 ➋
model = load_model('mnist_model.h5')
model.summary()
model.evaluate(x_test, y_test, verbose=2)
# 테스트셋에서 20번째 이미지 출력 ➌
plt.imshow(x_test[20], cmap="gray")
plt.show()
# 테스트셋의 20번째 이미지 클래스 분류 ➍
picks = [20]
predict = model.predict_classes(x_test[picks])
print("손글씨 이미지 예측값 : ", predict)