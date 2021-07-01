import glob

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.layers import Activation, Flatten, Lambda, Input, ELU
from tensorflow import keras
import math
import numpy as np
import cv2


def get_model():
    model = Sequential()
    # 5x5 Convolutional layers with stride of 2x2

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, 5, 2, name='conv1'))
    model.add(ELU(name='elu1'))
    model.add(Conv2D(36, 5, 2, name='conv2'))
    model.add(ELU(name='elu2'))
    model.add(Conv2D(48, 5, 2, name='conv3'))
    model.add(ELU(name='elu3'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, 3, 1, name='conv4'))
    model.add(ELU(name='elu4'))
    model.add(Conv2D(64, 3, 1, name='conv5'))
    model.add(ELU(name='elu5'))

    # Flatten before passing to the fully connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100, name='fc1'))
    model.add(Dropout(.5, name='do1'))
    model.add(ELU(name='elu6'))
    model.add(Dense(50, name='fc2'))
    model.add(Dropout(.5, name='do2'))
    model.add(ELU(name='elu7'))
    model.add(Dense(10, name='fc3'))
    model.add(Dropout(.5, name='do3'))
    model.add(ELU(name='elu8'))
    model.add(Dense(1, input_dim=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=100):
        self.x = []
        self.y = []
        for img_file in glob.glob("/home/adwiii/git/perception_fuzzing/src/images/**/*_edit.png"):
            self.x.append(img_file)
            self.y.append(1)  # edit class is 1
        for img_file in glob.glob("/home/adwiii/git/perception_fuzzing/src/images/**/*_orig.png"):
            self.x.append(img_file)
            self.y.append(0)  # orig class is 1
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_arr = np.array([cv2.imread(file_name) for file_name in batch_x], dtype=np.float32)
        y_arr = np.array(batch_y)
        # print(x_arr.shape, x_arr.dtype)
        # print(y_arr.shape, y_arr.dtype)
        return x_arr, y_arr


def main():
    with tf.device("gpu:0"):
        data_generator = DatasetGenerator()
        model = get_model()
        model.fit_generator(data_generator, steps_per_epoch=None, epochs=1, verbose=1,
                            callbacks=None, validation_data=None, validation_steps=None,
                            validation_freq=1, class_weight=None, max_queue_size=10,
                            workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
        model_name = 'realness_classifier'
        model.save_weights('{}.h5'.format(model_name))


if __name__ == '__main__':
    main()

