import tensorflow.keras as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Add, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from Data import Generator, GeneratorWithAugmentation

def ConstructModel(architecture, convlutionBlocks, denseLayer, filters, input_shape, numClasses):
    # TODO: Test this
    if architecture == 'ResNet':
        layer = lambda input, filters: ResDecreaseLayer(ResLayer(ResLayer(input, filters), filters), 2*filters)
    else:  # VGG
        layer = lambda input, filters: VGGLayer(input, filters)

    inp = Input(shape=input_shape)
    currentlayer = inp

    if architecture == 'ResNet':
        currentlayer = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format='channels_last')(currentlayer)
        currentlayer = BatchNormalization()(currentlayer)

    for i in range(convlutionBlocks):
        currentlayer = layer(currentlayer, filters)
        filters *= 2

    currentlayer = Flatten(data_format='channels_last')(currentlayer)
    for i in range(denseLayer):
        currentlayer = Dense(256, activation='relu')(currentlayer)
        currentlayer = Dropout(0.5)(currentlayer)

    result = Dense(numClasses, activation='softmax')(currentlayer)
    return Model(inputs=inp, outputs=result)

def CreateVGGModel(input_shape, num_classes):
    """Simple cnn model
    Data format - channels last
    Structure:
    2 conv 32, 3*3 | pool 2*2
    2 conv 64, 3*3 | pool 2*2
    dense 128 | dense num_classes, softmax"""
    inp = Input(shape=input_shape)

    conv_1 = VGGLayer(inp, 32)
    conv_2 = VGGLayer(conv_1, 64)
    dropout_1 = Dropout(0.25)(conv_2)

    flat = Flatten(data_format='channels_last')(dropout_1)
    dense_1 = Dense(128, activation='relu')(flat)
    dropout_2 = Dropout(0.5)(dense_1)

    result = Dense(num_classes, activation='softmax')(dropout_2)
    return Model(inputs=inp, outputs=result)

def CreateResNetModel(input_shape, num_classes):
    inp = Input(shape=input_shape)

    conv_0 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format='channels_last')(inp)
    bn_0 = BatchNormalization()(conv_0)

    res_1 = ResLayer(bn_0, 32)
    res_1 = ResLayer(res_1, 32)
    res_1 = ResLayer(res_1, 32)
    res_1 = ResLayer(res_1, 32)
    resDec_1 = ResDecreaseLayer(res_1, 64)

    res_2 = ResLayer(resDec_1, 64)
    res_2 = ResLayer(res_2, 64)
    res_2 = ResLayer(res_2, 64)
    res_2 = ResLayer(res_2, 64)
    avg_pool_1 = AveragePooling2D()(res_2)

    flat = Flatten(data_format='channels_last')(avg_pool_1)
    dense_1 = Dense(256, activation='relu')(flat)

    result = Dense(num_classes, activation='softmax')(dense_1)
    return Model(inputs=inp, outputs=result)

def VGGLayer(input, filters):
    conv_1 = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format='channels_last')(input)
    conv_2 = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format='channels_last')(conv_1)
    pool = MaxPooling2D((2, 2), data_format='channels_last')(conv_2)
    return pool

def ResLayer(input, filters):
    bn_1 = BatchNormalization()(input)
    a_1 = Activation('relu')(bn_1)
    conv_1 = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format='channels_last')(a_1)
    bn_2 = BatchNormalization()(conv_1)
    a_2 = Activation('relu')(bn_2)
    conv_2 = Conv2D(filters, (3, 3), padding="same", data_format='channels_last')(a_2)
    add = Add()([input, conv_2])
    return add

def ResDecreaseLayer(input, filters):
    conv_1 = Conv2D(filters, (3, 3), padding="same", strides=(2, 2), activation="relu", data_format='channels_last')(input)
    conv_2 = Conv2D(filters, (3, 3), padding="same", strides=(1, 1), data_format='channels_last')(conv_1)
    decConv = Conv2D(filters, (1, 1), padding="same", strides=(2, 2), data_format='channels_last')(input)
    add = Add()([decConv, conv_2])
    activation = Activation('relu')(add)
    return activation

def SaveModel(model, path):
    model.save(path)

def LoadModel(path):
    return load_model(path)

def CompileModel(model, loss=K.losses.categorical_crossentropy,
                 optimizer=K.optimizers.Adadelta(), metrics=['accuracy']):
    """Optimizers - https://keras.io/optimizers/
    Losses - https://keras.io/losses/"""
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

def FitModel(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10, useTensorboard=False, modelName='model'):
    callbacks = GetCallbacks(modelName, useTensorboard)
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))
    return history

def FitGenerator(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10, useTensorboard=False, modelName='model'):
    callbacks = GetCallbacks(modelName, useTensorboard)
    model.fit_generator(GeneratorWithAugmentation().flow(x_train, y_train, batch_size),
              epochs=epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))

def GetCallbacks(modelName, useTensorboard, batch_size=64):
    checkpoint = ModelCheckpoint(
        filepath='./log/'+modelName+'-{epoch:02d}-{val_acc:.2f}.hdf5',
        verbose=1,
        save_best_only=True,
        period=1)

    tbCallBack = TensorBoard(
        log_dir='./log', histogram_freq=1,
        write_graph=True,
        write_grads=True,
        batch_size=batch_size,
        write_images=True)
    callbacks = []
    if (useTensorboard):
        callbacks.append(tbCallBack)
    callbacks.append(checkpoint)
    return callbacks

def EvaluateModel(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
