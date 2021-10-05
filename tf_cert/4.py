# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import  train_test_split


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE

    with open('sarcasm.json', 'r', encoding='utf-8') as f:
        contents = f.read()
        json_data = json.loads(contents)

    for i in range(len(json_data)):
        x = json_data[i]["headline"]
        y = json_data[i]["is_sarcastic"]
        sentences.append(x)
        labels.append(y)

    sentences = np.array(sentences)
    labels = np.array(labels)
    labels = labels.reshape(26709, 1)

    token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(sentences)
    # print(token.word_index)

    x_data = token.texts_to_sequences(sentences)

    pad_x = pad_sequences(x_data, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    print(pad_x)
    print(pad_x.shape)  # (26709, 120)
    print(np.unique(pad_x))

    print(pad_x.shape)
    print(labels.shape)

    train_size = training_size / 26709

    x_train, x_test, y_train, y_test = train_test_split(pad_x, labels,
                                    train_size=train_size, shuffle=True)

    # 2. 모델
    model = tf.keras.Sequential([
        # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 3. 컴파일, 훈련
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.fit(x_train, y_train, epochs=10)

    # 4. 평가, 예측

    loss = model.evaluate(x_test, y_test)

    print("loss :", loss[0])        # loss : 0.7224743962287903
    print("acc :", loss[1])         # acc : 0.8022059798240662

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
