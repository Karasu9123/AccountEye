import tensorflow.keras as K
import Models as M
import Data
import cv2
import numpy as np

def ShowWrongPredict(model, x, y):
    for i in range(0, y.size):
        predict = model.predict(np.expand_dims(x[i], axis=0))
        predict = np.argmax(predict, axis=1)
        if (predict != y[i]):
            cv2.imshow(str(i), x[i])
            print(i, "| y: ", y[i]/2, " x: ", predict/2)
            cv2.waitKey(0)
    cv2.waitKey(0)

def experiment():
    # Settings
    train_part = 0.8
    batch_size = 128
    num_classes = 20
    epochs = 150
    img_rows, img_cols, channels = 48, 32, 3
    input_shape = (img_rows, img_cols, channels)
    #path = "sobelDS_3/"
    #file = 'labels.csv'
    csvPath = '../Images/Labels/All.csv'
    imagePath = '../Images/Resized/'

    # Load data
    (x_train, y_train_o), (x_test, y_test_o) = Data.load_data(csvPath, imagePath, img_rows, img_cols, channels=channels,
                                                              train_part=train_part)

    # Set type.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize to (0, 1) interval.
    x_train /= 255
    x_test /= 255

    # Print x.
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Converts y vector to binary class matrix.
    y_train = K.utils.to_categorical(y_train_o, num_classes)
    y_test = K.utils.to_categorical(y_test_o, num_classes)

    # Create model.
    model = M.LoadModel("/home/shared/AccountEye/Experiments/log/ResNet_All-130-0.89.hdf5")
    #model = M.CreateResNetModel(input_shape, num_classes)
    #model.summary()

    # Compile and fit.
    M.CompileModel(model)
    #M.FitGenerator(model, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=32, modelName='ResNet_Grey_All')
    M.EvaluateModel(model, x_test, y_test)
    #M.SaveModel(model, "first_test.h5")

    #ShowWrongPredict(model, x_test, y_test_o)

def networkSelection():
    # TODO: Test and write
    csvPath = '../Images/Labels/All.csv'
    imagePath = '../Images/Resized/'
    train_part = 0.8
    epochs = 100

    architecture = ['VGG', 'ResNet']
    convlutionBlocks = [2, 3, 4]
    denseLayer = [0, 1]
    filters = [8, 16, 32]
    numClasses = [('10_Classes', 10), ('20_Classes', 20)]
    data = [('label', 'image')]


    img_rows, img_cols, channels = 48, 32, 3
    (x_train, y_train_o), (x_test, y_test_o) = Data.load_data(csvPath, imagePath, img_rows, img_cols, channels=channels,
                                                              train_part=train_part)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    for arc in architecture:
        for block in convlutionBlocks:
            for dense in denseLayer:
                for filter in filters:
                    for classes in numClasses:
                        name = '{}_{}-blocks_{}-filters_{}-dense'.format(arc, block, filter, dense, classes[1])
                        print('-------------------------\n', name)
                        yTrain = K.utils.to_categorical(y_train_o, classes[1])
                        yTest = K.utils.to_categorical(y_test_o, classes[1])
                        model = M.ConstructModel(arc, block, dense, filter, (img_rows, img_cols, channels), classes[1])
                        M.CompileModel(model)
                        M.FitGenerator(model, x_train, yTrain, x_test, yTest, epochs=epochs, batch_size=32,
                                       modelName=name)

def main():
    experiment()

if __name__ == '__main__':
    main()