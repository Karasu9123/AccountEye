import tensorflow.keras as K
from tensorflow.keras.utils import multi_gpu_model
import Experiments.Models as M
import Experiments.Data as D
import ImagePreprocessing.ImageProcessing as IP
from sklearn.svm import SVC
import cv2
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.backend import clear_session
import gc



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


def ComparisonExperiment():
    trainPart = 0.6
    testPart = 0.3
    batchSize = 256
    numClasses = 10
    epochs = 10
    rows, cols, channels = 48, 32, 3
    input_shape = (rows, cols, channels)
    csvPaths = ['../Images/Labels/NewBalanced_Augmented.csv',
                '../Images/Labels/YouTube_Augmented.csv',
                '../Images/Labels/Clean_Augmented.csv']
    imgPaths = [('Original', '../Images/Preprocessed/Original/'),
     ('DoG', '../Images/Preprocessed/DoG/'),
     ('DoGThreshold', '../Images/Preprocessed/DoGThreshold/'),
     ('Sobel', '../Images/Preprocessed/Sobel/'),
     ('SobelThreshold', '../Images/Preprocessed/SobelThreshold/')]
    architectures = [("ResNet", M.CreateResNetModel),
                     ("VGG",    M.CreateVGGModel),
                     ("SVM",    SVC)]
    testCount = 3
    testResults = []

    for dsName, imgPath in imgPaths:
        for arcName, architecture in architectures:
            for testNum in range(testCount):
                # Load data
                (xTrain, yTrain), (xTest, yTest), (xValid, yValid) = D.LoadData(csvPaths, imgPath, rows, cols,
                    channels=channels, trainPart=trainPart, testPart=testPart, labelRow='10_Classes')

                if arcName == "SVM":
                    xTrainHOG = np.zeros((xTrain.shape[0], 540))
                    for i in range(0, xTrain.shape[0]):
                        xTrainHOG[i] = IP.HoG(xTrain[i]).flatten()

                    xTestHOG = np.zeros((xTest.shape[0], 540))
                    for i in range(0, xTest.shape[0]):
                        xTestHOG[i] = IP.HoG(xTest[i]).flatten()

                    xValidHOG = np.zeros((xValid.shape[0], 540))
                    for i in range(0, xValid.shape[0]):
                        xValidHOG[i] = IP.HoG(xValid[i]).flatten()

                    name = dsName + '_' + arcName + '_' + str(testNum) + '.sav'

                    # Create model.
                    model = architecture(max_iter=-1, verbose=True)

                    # Fit model.
                    model.fit(xTrainHOG, yTrain)

                    # Save model.
                    pickle.dump(model, open("./log/" + name, 'wb'))


                    accTrain = model.score(xTrainHOG, yTrain)
                    accTest = model.score(xTestHOG, yTest)
                    accValid = model.score(xValidHOG, yValid)
                    print("trainAcc = {},    testAcc = {},    validAcc = {}".format(accTrain, accTest, accValid))
                    testResults.append(name + ":    accuracy = " + str(accValid))
                else:
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
                    yTrain = K.utils.to_categorical(yTrain, numClasses)
                    yTest = K.utils.to_categorical(yTest, numClasses)
                    yValid = K.utils.to_categorical(yValid, numClasses)

                    # Create model.
                    model = architecture(input_shape, numClasses)
                    model = multi_gpu_model(model, gpus=2)
                    name = dsName + '_' + arcName + '_' + str(testNum)

                    # Compile and fit.
                    M.CompileModel(model)
                    M.FitModel(model, xTrain, yTrain, xTest, yTest, epochs=epochs, batch_size=batchSize, modelName=name)
                    loss, accuracy = M.EvaluateModel(model, xValid, yValid)
                    testResults.append(name + ":    loss = " + str(loss) +  ",  accuracy = " + str(accuracy))
    with open("testResults.txt", "w") as outfile:
        for result in testResults:
            outfile.write(result + '\n')


def EvaluateExperimentModels():
    validationDSDir = "../Images/Validation/Resized/"
    validationCSVDir = "../Images/Validation/validation_dataset.csv"
    rows, cols, channels = 48, 32, 3
    numClasses = 10
    imRow = 'Image_Name'
    labelRow = '10_Classes'
    preprocFuncs = [('Original', lambda a: a),
                   ('DoG', IP.PreprocDoG),
                   ('DoGThreshold', IP.PreprocDoGThreshold),
                   ('Sobel', IP.PreprocSobel),
                   ('SobelThreshold', IP.PreprocSobelThreshold)]
    validationResults = []

    df = pd.read_csv(validationCSVDir, index_col=None, header=0)
    labels = df.iloc[:][labelRow].values

    # Read images
    countRow = len(df.index)
    images = np.empty((countRow, rows, cols, channels), dtype=np.uint8)

    for index, row in df.iterrows():
        if channels == 3:
            images[index, ...] = cv2.imread(validationDSDir + row[imRow], 1)
        elif channels == 1:
            images[index, ...] = cv2.imread(validationDSDir + row[imRow], 0)[..., np.newaxis]

    print(images.shape[0], 'valid samples')
    y = K.utils.to_categorical(labels, numClasses)


    for preprocName, preproc in preprocFuncs:
        preprocessed = np.ndarray(images.shape, dtype=np.uint8)
        for idx, img in enumerate(images):
            preprocessed[idx] = preproc(img)

        # Eval CNN models
        x = preprocessed.astype('float32')
        x /= 255

        path = 'ExperimentModels/CNN/' + preprocName + '/'
        modelNames = IP.GetAllNames(path)

        for modelName in modelNames:
            model = M.LoadModel(path + modelName)
            M.CompileModel(model)
            print("\n" + modelName + ":")
            loss, accuracy = M.EvaluateModel(model, x, y)
            validationResults.append(modelName + ":    loss = " + str(loss) +  ",  accuracy = " + str(accuracy))
            del model
            clear_session()
            gc.collect()

        # Eval SVM models
        xHOG = np.zeros((preprocessed.shape[0], 540))
        for i in range(0, preprocessed.shape[0]):
            xHOG[i] = IP.HoG(preprocessed[i]).flatten()

        path = 'ExperimentModels/SVM/' + preprocName + '/'
        modelNames = IP.GetAllNames(path)

        for modelName in modelNames:
            model = pickle.load(open(path + modelName, 'rb'))
            accuracy = model.score(xHOG, labels)
            print(modelName + ":    validAcc = {}".format(accuracy))
            validationResults.append(modelName + ":    accuracy = " + str(accuracy))
            del model
            gc.collect()

        del preprocessed
        del x
        del xHOG

    with open("validationResults.txt", "w") as outfile:
        for result in validationResults:
            outfile.write(result + '\n')



def Main():
    EvaluateExperimentModels()



if __name__ == '__main__':
    Main()