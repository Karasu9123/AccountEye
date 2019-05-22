import tensorflow.keras as K
from tensorflow.keras.utils import multi_gpu_model
import Experiments.Models as M
import Experiments.Data as D
import cv2
import numpy as np
import matplotlib.pyplot as plt



def ShowWrongPredict(model, x, y):
    for i in range(0, y.size):
        predict = model.predict(np.expand_dims(x[i], axis=0))
        predict = np.argmax(predict, axis=1)
        if predict != y[i]:
            cv2.imshow(str(i), x[i])
            print(i, "| y: ", y[i] / 2, " x: ", predict / 2)
            cv2.waitKey(0)
    cv2.waitKey(0)


def Experiment():
    # Settings
    trainPart = 0.6
    testPart = 0.3
    batchSize = 256
    numClasses = 20
    epochs = 10
    rows, cols, channels = 48, 32, 1
    input_shape = (rows, cols, channels)
    csvPaths = ['../Images/Augmented/YouTube_Blur+Augmentation.csv',
                '../Images/Augmented/NewBalanced_Blur+Augmentation.csv',
                '../Images/Augmented/Clean_Blur+Augmentation.csv']
    imgPath = '../Images/Preproc/'

    # Load data
    (xTrain, yTrainO), (xTest, yTestO), (xValid, yValidO) = D.LoadData(csvPaths, imgPath, rows, cols, channels=channels,
                                                                       trainPart=trainPart, testPart=testPart,
                                                                       labelRow='20_Classes')

    # Set type.
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xValid = xValid.astype('float32')

    # Normalize to (0, 1) interval.
    xTrain /= 255
    xTest /= 255
    xValid /= 255

    # Print x.
    print('xTrain shape:', xTrain.shape)
    print(xTrain.shape[0], 'train samples')
    print(xTest.shape[0], 'test samples')
    print(xValid.shape[0], 'valid samples')

    # Converts y vector to binary class matrix.
    y_train = K.utils.to_categorical(yTrainO, numClasses)
    y_test = K.utils.to_categorical(yTestO, numClasses)
    y_valid = K.utils.to_categorical(yValidO, numClasses)

    # Create model.
    #model = M.LoadModel("ResNet_10_All-131-0.94.hdf5")
    model = M.CreateResNetModel(input_shape, numClasses)
    model = multi_gpu_model(model, gpus=2)
    # model.summary()

    # Compile and fit.
    M.CompileModel(model)
    M.FitModel(model, xTrain, y_train, xTest, y_test, epochs=epochs, batch_size=batchSize, modelName='ResNet_All_Blur')
    M.EvaluateModel(model, xValid, y_valid)

    # ShowWrongPredict(model, xTest, yTestO)


def NetworkSelection():
    # TODO: Test and write
    csvPath = '../Images/Labels/All.csv'
    imagePath = '../Images/Resized/'
    trainPart = 0.8
    epochs = 100

    architecture = ['ResNet', 'VGG']
    convlutionBlocks = [2, 3, 4]
    denseLayer = [0, 1]
    filters = [8, 16, 32]
    numClasses = [('20_Classes', 20)]
    data = [('label', 'image')]

    img_rows, img_cols, channels = 48, 32, 3
    (xTrain, yTrainO), (xTest, yTestO) = D.LoadData(csvPath, imagePath, img_rows, img_cols, channels=channels,
                                                    trainPart=trainPart)
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255
    xTest /= 255
    history = {}

    for arc in architecture:
        for block in convlutionBlocks:
            for dense in denseLayer:
                for filter in filters:
                    for classes in numClasses:
                        name = '{}_{}-blocks_{}-filters_{}-dense'.format(arc, block, filter, dense, classes[1])
                        print('-------------------------\n', name)
                        yTrain = K.utils.to_categorical(yTrainO, classes[1])
                        yTest = K.utils.to_categorical(yTestO, classes[1])
                        model = M.ConstructModel(arc, block, dense, filter, (img_rows, img_cols, channels), classes[1])
                        model = multi_gpu_model(model, gpus=2)
                        M.CompileModel(model)
                        history[name] = M.FitGenerator(model, xTrain, yTrain, xTest, yTest, epochs=epochs, batch_size=32,
                                       modelName=name)


def Main():
    # networkSelection()
    Experiment()



if __name__ == '__main__':
    Main()

