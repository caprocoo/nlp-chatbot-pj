# Chapter 6. 챗봇 엔진에 필요한 딥러닝 모델 (p.141 ~ p.)

# 목차
#   1. MNIST 분류모델 학습
#   2. 학습된 MNIST 분류 모델 파일 불러와 사용하기
#   3. 문장 분류를 위한 CNN 모델
#   4. 학습된 CNN 분류 모델 파일 불러와 사용하기
#   5. RNN 모델
#   6. LSTM 모델
#   7. 양방향 LSTM 모델


# 1. MNIST 분류모델 학습

# # 필요한 모듈 임포트 ➊
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense
#
# # MNIST 데이터셋 가져오기 ➋
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0 # 데이터 정규화
#
# # tf.data를 사용하여 데이터셋을 섞고 배치 만들기 ➌
# ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
# train_size = int(len(x_train) * 0.7) # 학습셋:검증셋 = 7:3
# train_ds = ds.take(train_size).batch(20)
# val_ds = ds.skip(train_size).batch(20)
#
# # MNIST 분류 모델 구성 ➍
# model = Sequential()
# model.add(Flatten(input_shape=(28, 28)))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# # 모델 생성 ➎
# model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# # model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
#
# # 모델 학습 ➏
# hist = model.fit(train_ds, validation_data=val_ds, epochs=10)
#
# # 모델 평가 ➐
# print('모델 평가')
# model.evaluate(x_test, y_test)
#
# # 모델 정보 출력 ➑
# model.summary()
#
# # 모델 저장 ➒
# model.save('mnist_model.h5')
#
# # 학습 결과 그래프 그리기
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()


# 2. 학습된 MNIST 분류 모델 파일 불러오기

# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# 
# # MNIST 데이터셋 가져오기 ➊
# _, (x_test, y_test) = mnist.load_data()
# x_test = x_test / 255.0 # 데이터 정규화
# # 모델 불러오기 ➋
# model = load_model('mnist_model.h5')
# model.summary()
# model.evaluate(x_test, y_test, verbose=2)
# # 테스트셋에서 20번째 이미지 출력 ➌
# plt.imshow(x_test[20], cmap="gray")
# plt.show()
# # 테스트셋의 20번째 이미지 클래스 분류 ➍
# picks = [20]
# predict = model.predict_classes(x_test[picks])
# print("손글씨 이미지 예측값 : ", predict)


# # 3. 문장 분류를 위한 CNN 모델
# # ChatbotData.csv 파일을 이용해 문장을 감정 클래스별로 분류하는 CNN 모델을 구현
#
# # 필요한 모듈 임포트
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import preprocessing
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
#
# # 데이터 읽어오기
# # Pandas 의 read_csv()를 이용해 csv 파일 읽기
# # Q(질문), label(감정) 데이터를 feature와 labels 리스트에 저장
# train_file = "../../ChatbotData.csv"
# data = pd.read_csv(train_file, delimiter=',')
# features = data['Q'].tolist()
# labels = data['label'].tolist()
#
# # 단어 인덱스 시퀀스 벡터
# # text_to_word_sequnce() 함수를 이용해 단어 시퀀스 생성 -> 단어 시퀀스란 단어 토큰들의 순차적 리스트를 의미
# # : ex) '3박 4일 놀러가고 싶다' 문장의 단어 시퀀스는 ['3박4일', '놀러가고', '싶다']
# # 단어 시퀀스를 말뭉치(corpus)에 저장
# corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
# tokenizer = preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(corpus)
#
# # 문장 내 모든 단어를 시퀀스 번호로 변환 -> 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터를 만들 것이기 때문
# sequences = tokenizer.texts_to_sequences(corpus)
# word_index = tokenizer.word_index
#
# # 여기서 고려할 것이 있다.
# # 시퀀스 번호로 만든 벡터는 문장의 길이가 제각각이기 때문에 벡터 크기가 다르다.
# # 시퀀스 번호로 변환된 전체 벡터 크기를 동일하게 맞춰야 한다.
# # 이 경우 MAX_SEQ_LEN 크기만큼 벡터크기를 맞추고, 이 크기보다 작은 벡터의 남는 공간에는 0으로 채우는 작업을 해야한다.
# # 이런 일련의 과정을 패딩처리라고 한다. -> 케라스에서는 pad_sequnces() 를 통해 손쉽게 처리할 수 있다.
# MAX_SEQ_LEN = 15 # 단어 시퀀스 벡터 크기(너무 크게 잡으면 자원낭비, 너무 작게 잡으면 입력 데이터의 손실 위험이 있음)
# padded_seq = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
#
#
# # 학습용, 검증용, 테스트용 데이터셋 생성
# ds = tf.data.Dataset.from_tensor_slices((padded_seq, labels))
# ds = ds.shuffle(len(features))
#
# # 학습셋:검증셋:테스트셋 = 7:2:1
# train_size = int(len(padded_seq) * 0.7) # 학습용 데이터
# val_size = int(len(padded_seq) * 0.2) # 검증용 데이터
# test_size = int(len(padded_seq) * 0.1) # 테스트 데이터
#
# train_ds = ds.take(train_size).batch(20)
# val_ds = ds.skip(train_size).take(val_size).batch(20)
# test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)
#
# # 하이퍼파라미터 설정
# dropout_prob = 0.5 # 오버피팅에 대비하기 위해 50%로 설정
# EMB_SIZE = 128
# EPOCH = 5
# VOCAB_SIZE = len(word_index) + 1 # 전체 단어 수
#
# # CNN 모델 정의
# input_layer = Input(shape=(MAX_SEQ_LEN,)) # shape 인자로 입력 노드에 들어올 데이터의 형상을 지정함.
# embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer) # 임베딩 계층 생성
# dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)
#
# # 임베딩 벡터를 합성곱 계층의 입력으로 받아 GlobalMaxPoll1D() 를 이용해 최대 풀링 연산을 수행
# conv1 = Conv1D(
#     filters=128,
#     kernel_size=3,
#     padding='valid',
#     activation=tf.nn.relu)(embedding_layer)
#
# pool1 = GlobalMaxPool1D()(conv1)
#
# conv2 = Conv1D(
#     filters=128,
#     kernel_size=4,
#     padding='valid',
#     activation=tf.nn.relu)(dropout_emb)
#
# pool2 = GlobalMaxPool1D()(conv2)
#
# conv3 = Conv1D(
#     filters=128,
#     kernel_size=5,
#     padding='valid',
#     activation=tf.nn.relu)(dropout_emb)
#
# pool3 = GlobalMaxPool1D()(conv3)
#
#
# # 3, 4, 5-gram 이후 합치기
# concat = concatenate([pool1, pool2, pool3]) # 각각 병렬로 처리된 합성곱 계층의 특징맵 결과를 하나로 묶어줌
#
# # Dense() 를 이용해서 128개의 출력 노드를 가지로 relu 활성화 함수를 사용하는 Dense 계층을 생성.
# # Dense 계층은 이전 계층에서 합성곱 연산과 맥스 풀링으로 나온 3개의 특징맵 데이터를 입력으로 받는다.
# hidden = Dense(128, activation=tf.nn.relu)(concat)
# dropout_hidden = Dropout(rate=dropout_prob)(hidden)
# logits = Dense(3, name='logits')(dropout_hidden) # 점수(score) : 3개의 점수가 출력되는데, 가장 큰 점수를 가진 노드 위치가 CNN 모델이 예측한 결과(class) 가 된다.
#
# # logits에서 나온 점수를 소프트맥스 계층을 통해 감정 클래스별 확률을 계산한다.
# # 클래스 분류 모델을 학습할 때 주로 손실값을 계산하는 함수로 sparse_categorical_crossentropy 를 사용한다.
# # -> 이때 크로스엔트로피 계산을 위해 확률값을 입력으로 사용해야 하는데 이를 위해 소프트맥스 계층이 필요하다.
# predictions = Dense(3, activation=tf.nn.softmax)(logits)
#
# # 모델 생성
# model = Model(inputs=input_layer, outputs=predictions)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy']) # 정확도를 확인
#
# # 모델 학습
# model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)
#
# # 모델 평가
# loss, accuracy = model.evaluate(test_ds, verbose=1) # verbose 1 : 모델 학습 시 진행과정을 상세히 보여줌 / 0 : 학습 과정 생략
# print('Accuracy: %f' % (accuracy * 100))
# print('loss: %f' % loss)
#
# # 모델 저장
# model.save('cnn_model.h5')
#
#
# # 위 예제에서 사용한 하이퍼파라미터값 (합성곱 필터 크기 및 개수, 단어 임베딩 벡터 크기, 에포크 값)이 꼭 정답은 아니다.
# # 만은 실험을 통해 최적의 하이퍼파라미터를 찾아야 한다.


# # 4. 학습된 CNN 분류 모델 파일 불러와 사용하기
# import tensorflow as tf
# import pandas as pd
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras import preprocessing
#
# # 데이터 읽어오기
# train_file = "../../ChatbotData.csv"
# data = pd.read_csv(train_file, delimiter=',')
# features = data['Q'].tolist()
# labels = data['label'].tolist()
#
# # 단어 인덱스 시퀀스 벡터
# corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
# tokenizer = preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(corpus)
# sequences = tokenizer.texts_to_sequences(corpus)
#
# MAX_SEQ_LEN = 15
# padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
#
# ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
# ds = ds.shuffle(len(features))
# test_ds = ds.take(2000).batch(20)
#
# model = load_model('cnn_model.h5')
# model.summary()
# model.evaluate(test_ds, verbose=2)
#
# print('단어 시퀀스 : ', corpus[10212])
# print('단어 인덱스 시퀀스 : ', padded_seqs[10212])
# print('문장 분류(정답) : ', labels[10212])
#
# picks = [10212]
# predict = model.predict(padded_seqs[picks])
# predict_class = tf.math.argmax(predict, axis=1)
# print('감정 예측 점수 : ', predict)
# print('감정 예측 클래스 : ', predict_class.numpy())


# # 5. RNN 모델
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, LSTM, SimpleRNN
#
#
# def split_sequence(sequence, step):
#     x, y = list(), list()
#
#     for i in range(len(sequence)):
#         end_idx = i + step
#         if end_idx > len(sequence) - 1:
#             break
#
#         seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
#         x.append(seq_x)
#         y.append(seq_y)
#
#     return np.array(x), np.array(y)
#
#
# # sin 함수 학습 데이터
# x = [i for i in np.arange(start=-10, stop=10, step=0.1)]
# train_y = [np.sin(i) for i in x]
#
# # 하이퍼 파라미터
# n_timesteps = 15
# n_features = 1
#
# # 시퀀스 나누기
# # train_x.shape => (samples, timesteps)
# # train_y.shape => (samples)
# train_x, train_y = split_sequence(train_y, step=n_timesteps)
# print('shape x:{} / y:{}'.format(train_x.shape, train_y.shape))
#
# # RNN 입력 벡터 크기를 맞추기 위해 벡터 차원 크기 변경
# # reshape from [samples, timesteps] into [samples, timesteps, features]
# train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
# print('train_x.shape = {}'.format(train_x.shape))
# print('train_y.shape = {}'.format(train_y.shape))
#
# # RNN 모델 정의
# model = Sequential()
# model.add(SimpleRNN(units=100, return_sequences=False, input_shape=(n_timesteps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
#
# # 모델 학습
# np.random.seed(0)
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(
#     monitor='loss',
#     patience=5,
#     mode='auto'
# )
# history = model.fit(train_x, train_y, epochs=1000, callbacks=[early_stopping])
#
# # loss 그래프 생성
# plt.plot(history.history['loss'], label='loss')
# plt.legend(loc='upper right')
# plt.show()
#
# # 테스트 데이터셋 생성
# test_x = np.arange(10, 20, 0.1)
# calc_y = np.cos(test_x)
#
# # RNN 모델 예측 및 로그 저장
# test_y=calc_y[:n_timesteps]
# for i in range(len(test_x) - n_timesteps):
#     net_input = test_y[i : i + n_timesteps]
#     net_input = net_input.reshape((1, n_timesteps, n_features))
#     train_y = model.predict(net_input, verbose=0)
#     print(test_y.shape, train_y.shape, i, i+n_timesteps)
#     test_y = np.append(test_y, train_y)
#
# # 예측 결과 그래프 그리기
# plt.plot(test_x, calc_y, label='ground truth', color='orange')
# plt.plot(test_x, test_y, label='predictions', color='blue')
#
# plt.legend(loc='upper left')
# plt.ylim(-2, 2)
# plt.show()


# 6. LSTM 모델
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LSTM


def split_sequence(sequence, step):
    x, y = list(), list()

    for i in range(len(sequence)):
        end_idx = i + step
        if end_idx > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


# sin 함수 학습 데이터
x = [i for i in np.arange(start=-10, stop=10, step=0.1)]
train_y = [np.sin(i) for i in x]

# 하이퍼 파라미터
n_timesteps = 15
n_features = 1

# 시퀀스 나누기
# train_x.shape => (samples, timesteps)
# train_y.shape => (samples)
train_x, train_y = split_sequence(train_y, step=n_timesteps)
print('shape x:{} / y:{}'.format(train_x.shape, train_y.shape))

# LSTM 입력 벡터 크기를 맞추기 위해 벡터 차원 크기 변경
# reshape from [samples, timesteps] into [samples, timesteps, features]
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)
print('train_x.shape = {}'.format(train_x.shape))
print('train_y.shape = {}'.format(train_y.shape))

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=10, return_sequences=False, input_shape=(n_timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 모델 학습
np.random.seed(0)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    mode='auto'
)
history = model.fit(train_x, train_y, epochs=1000, callbacks=[early_stopping])

# loss 그래프 생성
plt.plot(history.history['loss'], label='loss')
plt.legend(loc='upper right')
plt.show()

# 테스트 데이터셋 생성
test_x = np.arange(10, 20, 0.1)
calc_y = np.cos(test_x)

# RNN 모델 예측 및 로그 저장
test_y = calc_y[:n_timesteps]
for i in range(len(test_x) - n_timesteps):
    net_input = test_y[i: i + n_timesteps]
    net_input = net_input.reshape((1, n_timesteps, n_features))
    train_y = model.predict(net_input, verbose=0)
    print(test_y.shape, train_y.shape, i, i + n_timesteps)
    test_y = np.append(test_y, train_y)

# 예측 결과 그래프 그리기
plt.plot(test_x, calc_y, label='ground truth', color='orange')
plt.plot(test_x, test_y, label='predictions', color='blue')

plt.legend(loc='upper left')
plt.ylim(-2, 2)
plt.show()

# 7. 양방향 LSTM 모델
import numpy as np
from random import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, LSTM, TimeDistributed


# 시퀀스 생성
def get_sequence(n_timesteps):
    # 0~1 사이의 랜덤 시퀀스 생성
    X = np.array([random() for _ in range(n_timesteps)])

    # 클래스 분류 기준
    limit = n_timesteps / 4.0

    # 누적합 시퀀스에서 클래스 결정
    # 누적합 항목이 limit 보다 작은 경우 0, 아닌 경우 1로 분류
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    # LSTM 입력을 위해 3차원 텐서 형태로 변경
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y

# 하이퍼파라미터 정의
n_units = 20
n_timesteps = 4

# 양방향 LSTM 모델 정의
model = Sequential()
model.add(Bidirectional(LSTM(n_units, return_sequences=True, input_shape=(n_timesteps,1))))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# 에포크마다 학습 데이터를 생성해서 학습
for epoch in range(1000):
    X, y = get_sequence(n_timesteps)
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)

# 모델 평가
X, y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('실젯값 : ', y[0, i], '예측값 : ', yhat[0, i])




# Chapter .  (p. ~ p.)

# 목차
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.
