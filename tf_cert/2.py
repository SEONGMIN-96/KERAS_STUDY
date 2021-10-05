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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # YOUR CODE HERE
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D
    from tensorflow.keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train/255, x_test/255
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    input = Input(shape=(28, 28))
    xx = Conv1D(64, 2, activation='relu')(input)
    xx = MaxPooling1D(2)(xx)
    xx = Conv1D(32, 2, activation='relu')(xx)
    xx = MaxPooling1D(2)(xx)
    xx = Conv1D(32, 2, activation='relu')(xx)
    xx = MaxPooling1D(2)(xx)
    xx = Conv1D(16, 2, activation='relu')(xx)
    xx = Flatten()(xx)
    xx = Dense(16, activation='relu')(xx)
    output = Dense(10, activation='softmax')(xx)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=21, batch_size=128, validation_split=0.05)
    loss = model.evaluate(x_test, y_test)
    print('loss = ', loss[0])
    print('acc = ', loss[1])
    return model
# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")