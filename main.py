from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import Row
from bokeh.embed import components

np.random.seed(1)


def preprocess(X, Y):
    encoder = LabelBinarizer()
    Y_ = encoder.fit_transform(Y)
    X_ = preprocessing.scale(X)
    return X_, Y_


def neuralNet():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def plot(hist, plot_width = 500, plot_height = 500):
    x_acc = np.arange(0, len(hist.history['acc']))
    x_loss = np.arange(0, len(hist.history['loss']))

    p = figure(toolbar_location=None, tools='hover',
               plot_width=plot_width, plot_height=plot_height)
    p.line(x=x_loss, y=hist.history['loss'], line_width=3,
           alpha=0.75, color='#3498DB', legend='Loss')
    p.line(x=x_loss, y=hist.history['val_loss'], line_width=3,
           alpha=0.75, color='#1ABC9C', legend='Validation Loss')
    p.xaxis.axis_label = 'EPOCH'
    p.yaxis.axis_label = 'Loss'
    p.axis.minor_tick_line_color = None
    p.title.text = "Model Loss"
    p.title.align = 'center'

    j = figure(toolbar_location=None, tools='hover',
               plot_width=plot_width, plot_height=plot_height)
    j.line(x=x_acc, y=hist.history['acc'], line_width=3, alpha=0.75,
           color='#3498DB', legend='Model Accuracy')
    j.line(x=x_acc, y=hist.history['val_acc'], line_width=3, alpha=0.75,
           color='#1ABC9C', legend='Validation Accuracy')
    j.xaxis.axis_label = 'EPOCH'
    j.yaxis.axis_label = 'Accuracy'
    j.legend.location = 'bottom_right'
    j.axis.minor_tick_line_color = None
    j.title.text = 'Model Accuracy'
    j.title.align = 'center'

    return p, j


if __name__ == "__main__":
    # d = load_iris()
    # X, Y = preprocess(d.data, d.target)
    # model = neuralNet(X, Y)
    # history = model.fit(X, Y, validation_split=0.2, epochs=1000, batch_size=64)
    # plot(history)
    dataAugmentation = ImageDataGenerator(
        rescale=1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True)
    trainGenerator = dataAugmentation.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=2,
        class_mode='binary')
    validationGenerator = dataAugmentation.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=2,
        class_mode='binary')
    model = neuralNet()
    #model = load_model('model.h5')
    history = model.fit_generator(
        trainGenerator,
        steps_per_epoch=512,
        epochs=10,
        validation_data=validationGenerator,
        validation_steps=32)
    model.save('model.h5')
    p, j = plot(history)
    show(Row(p, j))
