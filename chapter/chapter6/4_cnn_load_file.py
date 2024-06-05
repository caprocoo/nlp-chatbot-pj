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



# 4. 학습된 CNN 분류 모델 파일 불러와 사용하기
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

# 데이터 읽어오기
train_file = "../../ChatbotData.csv"
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

# 단어 인덱스 시퀀스 벡터
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))
test_ds = ds.take(2000).batch(20)

model = load_model('cnn_model.h5')
model.summary()
model.evaluate(test_ds, verbose=2)

print('단어 시퀀스 : ', corpus[10212])
print('단어 인덱스 시퀀스 : ', padded_seqs[10212])
print('문장 분류(정답) : ', labels[10212])

picks = [10212]
predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)
print('감정 예측 점수 : ', predict)
print('감정 예측 클래스 : ', predict_class.numpy())