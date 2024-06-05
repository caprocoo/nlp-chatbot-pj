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



# 3. 문장 분류를 위한 CNN 모델
# ChatbotData.csv 파일을 이용해 문장을 감정 클래스별로 분류하는 CNN 모델을 구현

# 필요한 모듈 임포트
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

# 데이터 읽어오기
# Pandas 의 read_csv()를 이용해 csv 파일 읽기
# Q(질문), label(감정) 데이터를 feature와 labels 리스트에 저장
train_file = "../../ChatbotData.csv"
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

# 단어 인덱스 시퀀스 벡터
# text_to_word_sequnce() 함수를 이용해 단어 시퀀스 생성 -> 단어 시퀀스란 단어 토큰들의 순차적 리스트를 의미
# : ex) '3박 4일 놀러가고 싶다' 문장의 단어 시퀀스는 ['3박4일', '놀러가고', '싶다']
# 단어 시퀀스를 말뭉치(corpus)에 저장
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)

# 문장 내 모든 단어를 시퀀스 번호로 변환 -> 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터를 만들 것이기 때문
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index

# 여기서 고려할 것이 있다.
# 시퀀스 번호로 만든 벡터는 문장의 길이가 제각각이기 때문에 벡터 크기가 다르다.
# 시퀀스 번호로 변환된 전체 벡터 크기를 동일하게 맞춰야 한다.
# 이 경우 MAX_SEQ_LEN 크기만큼 벡터크기를 맞추고, 이 크기보다 작은 벡터의 남는 공간에는 0으로 채우는 작업을 해야한다.
# 이런 일련의 과정을 패딩처리라고 한다. -> 케라스에서는 pad_sequnces() 를 통해 손쉽게 처리할 수 있다.
MAX_SEQ_LEN = 15 # 단어 시퀀스 벡터 크기(너무 크게 잡으면 자원낭비, 너무 작게 잡으면 입력 데이터의 손실 위험이 있음)
padded_seq = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')


# 학습용, 검증용, 테스트용 데이터셋 생성
ds = tf.data.Dataset.from_tensor_slices((padded_seq, labels))
ds = ds.shuffle(len(features))

# 학습셋:검증셋:테스트셋 = 7:2:1
train_size = int(len(padded_seq) * 0.7) # 학습용 데이터
val_size = int(len(padded_seq) * 0.2) # 검증용 데이터
test_size = int(len(padded_seq) * 0.1) # 테스트 데이터

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# 하이퍼파라미터 설정
dropout_prob = 0.5 # 오버피팅에 대비하기 위해 50%로 설정
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1 # 전체 단어 수

# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,)) # shape 인자로 입력 노드에 들어올 데이터의 형상을 지정함.
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer) # 임베딩 계층 생성
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 임베딩 벡터를 합성곱 계층의 입력으로 받아 GlobalMaxPoll1D() 를 이용해 최대 풀링 연산을 수행
conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(embedding_layer)

pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)

pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)

pool3 = GlobalMaxPool1D()(conv3)


# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3]) # 각각 병렬로 처리된 합성곱 계층의 특징맵 결과를 하나로 묶어줌

# Dense() 를 이용해서 128개의 출력 노드를 가지로 relu 활성화 함수를 사용하는 Dense 계층을 생성.
# Dense 계층은 이전 계층에서 합성곱 연산과 맥스 풀링으로 나온 3개의 특징맵 데이터를 입력으로 받는다.
hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden) # 점수(score) : 3개의 점수가 출력되는데, 가장 큰 점수를 가진 노드 위치가 CNN 모델이 예측한 결과(class) 가 된다.

# logits에서 나온 점수를 소프트맥스 계층을 통해 감정 클래스별 확률을 계산한다.
# 클래스 분류 모델을 학습할 때 주로 손실값을 계산하는 함수로 sparse_categorical_crossentropy 를 사용한다.
# -> 이때 크로스엔트로피 계산을 위해 확률값을 입력으로 사용해야 하는데 이를 위해 소프트맥스 계층이 필요하다.
predictions = Dense(3, activation=tf.nn.softmax)(logits)

# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # 정확도를 확인

# 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가
loss, accuracy = model.evaluate(test_ds, verbose=1) # verbose 1 : 모델 학습 시 진행과정을 상세히 보여줌 / 0 : 학습 과정 생략
print('Accuracy: %f' % (accuracy * 100))
print('loss: %f' % loss)

# 모델 저장
model.save('cnn_model.h5')


# 위 예제에서 사용한 하이퍼파라미터값 (합성곱 필터 크기 및 개수, 단어 임베딩 벡터 크기, 에포크 값)이 꼭 정답은 아니다.
# 만은 실험을 통해 최적의 하이퍼파라미터를 찾아야 한다.