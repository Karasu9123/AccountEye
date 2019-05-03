import numpy as np
import pandas as pd
import cv2
import random
import tensorflow.keras.preprocessing.image



def LoadData(csvPaths, imgPath, rows, cols, channels = 3, trainPart = 0.6, testPart = 0.3, balanceDS = True, imRow = 'Image_Name', labelRow = '20_Classes'):
    """Channels: 3 for RGB, 1 for GS(gray scale)"""
    li = []
    for i in range(len(csvPaths)):
        df = pd.read_csv(csvPaths[i], index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    if balanceDS:
        df = CropDS(df, 3 if labelRow == '20_Classes' else 1)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Read images
    countRow = len(df.index)
    x = np.empty((countRow, rows, cols, channels), dtype=np.uint8)

    for index, row in df.iterrows():
        if channels == 3:
            x[index, ...] = cv2.imread(imgPath + row[imRow], 1)
        elif channels == 1:
            x[index, ...] = cv2.imread(imgPath + row[imRow], 0)[..., np.newaxis]

    # Split data
    train = int(trainPart * countRow)
    test = int((trainPart + testPart) * countRow)
    xTrain, xTest, xValid = x[:train, ...], x[train:test, ...], x[test:, ...]
    yTrain, yTest, yValid = df.iloc[:train][labelRow].values, df.iloc[train:test][labelRow].values, df.iloc[test:][labelRow].values

    return (xTrain, yTrain), (xTest, yTest), (xValid, yValid)


def CountNumbers(csvPath):
    df = pd.read_csv(csvPath)
    count = []
    for col in df.columns:
        count.append(df.groupby([col])[[col]].count())

    return count


def MinCount(df, column):
    col = df.columns[column]
    count = df.groupby([col])[[col]].count()

    return count.min(0)[0]


def CropDS(df, column = 1):
    minimal = MinCount(df, column) # количество изображений класса с найменьшим количеством примеров
    col = df.columns[column]
    li = []
    rng = 10 if column == 1 else 20
    for i in range(rng):
        li.append(df[df[col] == i].sample(frac=1).head(minimal))
    df = pd.concat(li, axis=0, ignore_index=True)

    return df


def Generator(features, labels, batchSize):
    # FIXME: It's not work! It's just a sample!
    batchFeatures = np.zeros((batchSize, 64, 64, 3))
    batchLabels = np.zeros((batchSize, 1))
    while True:
        for i in range(batchSize):
            # choose random index in features
            index = random.choice(len(features), 1)
            batchFeatures[i] = features[index] #some_processing(features[index])
            batchLabels[i] = labels[index]
        yield batchFeatures, batchLabels


def GeneratorWithAugmentation():
    return tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.01, 1.2),
        fill_mode='nearest')
