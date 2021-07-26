from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.layers.core import Flatten

# 1. 데이터

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 햇어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5) # or post
print(pad_x)
print(pad_x.shape) # (13, 5)

print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27]

# 원핫인코딩하면 (13, 5) -> (13, 5, 27)
# 옥스포드? (13, 5, 1000000) -> 6500만개

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Conv1D

model = Sequential()
                # 단어사전의 개수  아웃풋 노드 개수  단어수, 길이
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
model.add(Embedding(28, 77))
model.add(Conv1D(32, 2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308       inputdim*outputdim
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,973
Trainable params: 5,973
Non-trainable params: 0
'''

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x, labels, epochs=100, batch_size=8)

# 4. 평가, 예측

acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)