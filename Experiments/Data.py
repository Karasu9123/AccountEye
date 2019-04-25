import numpy as np
import pandas as pd
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(csvPaths, imgPath, rows, cols, channels=3, train_part=0.6, test_part=0.3, imRow='Image_Name', labelRow='20_Classes'):
    """Channels: 3 for RGB, 1 for GS(gray scale)"""
    li = []
    for i in range(len(csvPaths)):
        df = pd.read_csv(csvPaths[i], index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df = cropDS(df, 3 if labelRow=='20_Classes' else 1)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Read images
    count_row = len(df.index)
    x = np.empty((count_row, rows, cols, channels), dtype=np.uint8)

    for index, row in df.iterrows():
        # print(imgPath + row[imRow])
        if channels == 3:
            x[index, ...] = cv2.imread(imgPath + row[imRow], 1)
        elif channels == 1:
            x[index, ...] = cv2.imread(imgPath + row[imRow], 0)[..., np.newaxis]

    # Split data
    train = int(train_part * count_row)
    test = int((train_part + test_part)* count_row)
    x_train, x_test, x_valid = x[:train, ...], x[train:test, ...], x[test:, ...]
    y_train, y_test, y_valid = df.iloc[:train][labelRow].values, df.iloc[train:test][labelRow].values, df.iloc[test:][labelRow].values
    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)

def countNumbers(csvPath):
    df = pd.read_csv(csvPath)
    count = []
    for col in df.columns:
        count.append(df.groupby([col])[[col]].count())
    return count

def minCount(df, column):
    col = df.columns[column]
    count = df.groupby([col])[[col]].count()
    return count.min(0)[0]

def cropDS(df, column = 1):
    min = minCount(df, column) # количество изображений класса с найменьшим количеством примеров
    col = df.columns[column]
    li = []
    rng = 10 if column == 1 else 20
    for i in range(rng):
        li.append(df[df[col] == i].sample(frac=1).head(min))
    df = pd.concat(li, axis=0, ignore_index=True)
    return df

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