from __future__ import print_function

from keras import backend
from keras.layers import Dense, Flatten
from keras.layers import Input, Convolution3D, BatchNormalization, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam

backend.set_image_dim_ordering('th')  # Theano dimension ordering in this code

num_filters = [16, 32, 64, 128, 256, 1028]


def get_simp3d(lr=4e-5):
    inputs = Input((1, 64, 64, 64))
    conv1 = Convolution3D(num_filters[0], 9, 9, 9, activation='relu', border_mode='valid')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution3D(num_filters[0], 3, 3, 3, activation='relu', border_mode='valid')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution3D(num_filters[0], 5, 5, 5, activation='relu', border_mode='valid')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Convolution3D(num_filters[1], 3, 3, 3, activation='relu', border_mode='valid')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution3D(num_filters[1], 3, 3, 3, activation='relu', border_mode='valid')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Convolution3D(num_filters[2], 3, 3, 3, activation='relu', border_mode='valid')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution3D(num_filters[2], 3, 3, 3, activation='relu', border_mode='valid')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Convolution3D(num_filters[3], 3, 3, 3, activation='relu', border_mode='valid')(conv3)
    conv4 = BatchNormalization()(conv4)

    flat = Flatten()(conv4)

    dense6 = Dense(output_dim=256, activation='relu')(flat)
    dense6 = BatchNormalization()(dense6)
    dense7 = Dense(output_dim=2, activation='softmax')(dense6)

    model = Model(input=inputs, output=dense7)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

    return model
