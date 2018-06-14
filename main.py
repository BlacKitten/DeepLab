import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Dropout
from keras.regularizers import l2
from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

np.random.seed(1)


def preprocess(X, Y):
    encoder = LabelBinarizer()
    Y_ = encoder.fit_transform(Y)
    X_ = preprocessing.scale(X)
    return X_, Y_


def neuralNet(X, Y):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=X.shape[1], kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', input_dim=X.shape[1], kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    d = load_iris()
    X, Y = preprocess(d.data, d.target)
    model = neuralNet(X, Y)
    history = model.fit(X, Y, validation_split=0.2, epochs=1000, batch_size=64)
    plot(history)
