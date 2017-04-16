from __future__ import print_function

import argparse
import os
from glob import glob

import numpy as np
from keras import backend as bck_end
from keras.layers import Dense, Flatten
from keras.layers import Input, Convolution3D, BatchNormalization, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam

from preprocessing import ImageDataGenerator
from utils import ModelCheckpointS3

# for reproducibility:
np.random.seed(2017)

bck_end.set_image_dim_ordering('th')  # Theano dimension ordering in this code

num_filters = [16, 32, 64, 128, 256, 1028]


def get_simp3d(lr):
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


def main(arguments):

    n_epoch_0 = arguments.n_epochs

    print('begin training, saving to weights/{}.h5, upload to s3 = {}'.format(arguments.weight_filename, arguments.s3))

    seed = np.random.randint(1002)
    train_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True,
                                       zoom_range=0,
                                       rotation_range=arguments.rotation,
                                       shear_range=arguments.shear,
                                       height_shift_range=0.1,
                                       width_shift_range=0.1,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       depth_flip=True)

    val_datagen = ImageDataGenerator(samplewise_center=True,
                                     samplewise_std_normalization=True)

    with open('patient_list.csv', mode='rb') as f:
        import csv
        wr = csv.reader(f, quoting=csv.QUOTE_ALL)
        patient_list = list(wr)[0]

    # kf = KFold(n_splits=10, random_state=100)
    bags = []
    split_ratio = 0.9
    for i in range(arguments.total_folds):
        np.random.shuffle(patient_list)
        nbr_train = int(len(patient_list)*split_ratio)
        train_patients = patient_list[:nbr_train]
        val_patients = patient_list[nbr_train:]
        bags += [(train_patients, val_patients)]

    data_base = '../data/luna/preprocessed_3d/'
    folders = ['cube_scans_neg', 'cube_scans_pos']

    y = []
    x_filenames = []
    for n, folder in enumerate(folders):

        scanfile_list = glob(data_base + folder + '/' + "*.npy")
        if n == 0:
            np.random.seed(568)
            np.random.shuffle(scanfile_list)
            scanfile_list = scanfile_list[:4500]
        y += [n] * len(scanfile_list)
        x_filenames += scanfile_list

    y_truth = np.array(y)

    false_pos_list = glob(data_base + 'false_pos/' + "*.npy")
    false_pos_y_truth = [0]*len(false_pos_list)

    true_pos_list = glob(data_base + 'true_pos/' + "*.npy")
    true_pos_y_truth = [1] * len(true_pos_list)

    print('total positive samples: {}'.format(y_truth.sum()))
    print('total false pos samples: {}'.format(len(false_pos_list)))
    print('total true pos samples: {}'.format(len(true_pos_list)))

    for n_fold, (train_patients, val_patients) in enumerate(bags):

        assert (len(set(val_patients).intersection(train_patients)) == 0)
        print(len(train_patients), len(val_patients))

        train_scanfile_list = [scanfile for scanfile in x_filenames
                               if os.path.basename(scanfile).split('_')[-1][:-4] in train_patients]

        train_scanfile_list += false_pos_list
        train_scanfile_list += true_pos_list

        train_scanfile_truth = [y for y, scanfile in zip(y_truth, x_filenames)
                                         if os.path.basename(scanfile).split('_')[-1][:-4] in train_patients]

        train_scanfile_truth += false_pos_y_truth
        train_scanfile_truth += true_pos_y_truth

        train_scanfile_truth = np.array(train_scanfile_truth)

        val_scanfile_list = [scanfile for scanfile in x_filenames
                             if os.path.basename(scanfile).split('_')[-1][:-4] in val_patients]

        val_scanfile_truth = np.array([y for y, scanfile in zip(y_truth, x_filenames)
                                       if os.path.basename(scanfile).split('_')[-1][:-4] in val_patients])

        assert (len(set(train_scanfile_list).intersection(val_scanfile_list)) == 0)

        print(len(train_scanfile_list), len(val_scanfile_list))
        train_generator = train_datagen.flow_from_filenames_3d_class(train_scanfile_list, train_scanfile_truth,
                                                                     seed=seed, batch_size=16)
        val_generator = val_datagen.flow_from_filenames_3d_class(val_scanfile_list, val_scanfile_truth,
                                                                 seed=seed, batch_size=16)

        (images_test, y_test) = train_generator.next()
        print(images_test.shape, y_test.shape, images_test.dtype, y_test.dtype)
        print(y_test.max(), y_test.dtype)

        if n_fold >= 3:

            if n_fold == 0:
                arguments.n_epochs = 50
            else:
                arguments.n_epochs = n_epoch_0

            model_3d = get_simp3d(arguments.lr)

            if arguments.load_weights:
                # noinspection PyBroadException
                try:
                    model_3d.load_weights('weights/{}_{}.h5'.format(arguments.load_weights, n_fold))
                    print(' loading weights/{}_{}.h5'.format(arguments.load_weights, n_fold))
                except Exception:
                    print('no weights found')

            # autosave best Model

            if not os.path.exists('weights'):
                os.makedirs('weights')

            best_model_file = "../weights/{}_{}.h5".format(arguments.weight_filename, n_fold)

            best_model = ModelCheckpointS3(best_model_file, monitor='val_categorical_crossentropy', verbose=1,
                                           save_best_only=True, upload_to_s3=arguments.s3)

            model_3d.fit_generator(train_generator, validation_data=val_generator,
                                   nb_val_samples=len(val_scanfile_list),
                                   samples_per_epoch=len(train_scanfile_list),
                                   nb_epoch=arguments.n_epochs, callbacks=[best_model])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train some 3d neurons')
    parser.add_argument('-s3', action='store_true')
    parser.add_argument('--rotation', type=float, default=15.0)
    parser.add_argument('--shear', type=float, default=0.1)
    parser.add_argument('--weight_filename', type=str, default='weights_3dconv')
    parser.add_argument('--total_folds', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--load_weights', type=str)

    args = parser.parse_args()

    main(args)
