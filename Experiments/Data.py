import numpy as np
import pandas as pd
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#def load_data(path, csvFileName, rows, cols, channels = 3, train_part=0.8, imRow='Image_name', labelRow='10_Classes'):
  #  pass

def load_data(csvPath, imagePath, rows, cols, channels=3, train_part=0.8, imRow='Image_Name', labelRow='20_Classes'):
    """Channels: 3 for RGB, 1 for GS(gray scale)"""
    # TODO Try remove `rows` & `cols` maybe
    #   Think about `imCol` & `labelCol`

    df = pd.read_csv(csvPath)

    # Remove nulls
    df.dropna()

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Read images
    count_row = len(df.index)
    x = np.empty((count_row, rows, cols, channels), dtype=np.uint8)

    for index, row in df.iterrows():
        if (channels == 3):
            x[index, ...] = cv2.imread(imagePath + row[imRow], 1)
        elif (channels == 1):
            x[index, ...] = cv2.imread(imagePath + row[imRow], 0)[..., np.newaxis]

    # Split data
    separator = int(train_part * count_row)
    x_train, x_test = x[:separator, ...], x[separator:, ...]
    y_train, y_test = df.iloc[:separator][labelRow].values, df.iloc[separator:][labelRow].values

    return (x_train, y_train), (x_test, y_test)

def Generator(features, labels, batch_size):
    # FIXME: It's not work! It's just a sample!
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(len(features), 1)
            batch_features[i] = features[index] #some_processing(features[index])
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels

def GeneratorWithAugmentation():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.01, 1.2),
        fill_mode='nearest')


"""Кривой вариант, на память
def load_data():
    fpaths = []
    y = []

    # read fpaths and labels from csv
    with open(path + "labels.csv") as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            fpaths.append(path + row["pic_name"])
            y.append(int(row["digit"]))

    # read images, and cast x, y to ndarray
    x = np.empty((len(fpaths), channels, img_rows, img_cols), dtype=np.uint8)
    y = np.asarray(y)
    for i, fpath in enumerate(fpaths):
        img = cv2.imread(fpath, 1)
        x[i, ...] = img.transpose(2, 0, 1)

    # shuffle and split
    indices = np.random.permutation(x.shape[0])
    separator = int(train_part * x.shape[0])
    training_idx, test_idx = indices[:separator], indices[separator:]
    x_train, x_test = x[training_idx, :], x[test_idx, :]
    y_train, y_test = y[training_idx], y[test_idx]

    # return couple
    return (x_train, y_train), (x_test, y_test)
"""