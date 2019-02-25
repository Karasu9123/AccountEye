import cv2
import numpy as np
import ImagePreprocessing.ImageProcessing as IP

def GetCam(idCam=0):
    return cv2.VideoCapture(idCam)

def GetFoto(cam, channels=3):
    ret, foto = cam.read()
    foto = IP.preproc(foto)
    #foto = cv2.imread("meter.jpg", 1)
    #if channels == 1:
    foto = cv2.cvtColor(foto, cv2.COLOR_GRAY2BGR)
    return foto

def main():
    import json
    from tensorflow.keras.models import load_model

    settingsPath = 'settings.json'
    modelPath = '/home/shared/AccountEye/Experiments/ResNet_10_All-131-0.94.hdf5'
    showOriginal = True
    showCrop = False
    channels = 3

    # Load Settings
    cam = GetCam()
    tempFoto = GetFoto(cam)
    with open(settingsPath) as f:
        settings = json.load(f)
    x0 = int(settings["region"]["x0"] * tempFoto.shape[1])
    x1 = int(settings["region"]["x1"] * tempFoto.shape[1])
    y0 = int(settings["region"]["y0"] * tempFoto.shape[0])
    y1 = int(settings["region"]["y1"] * tempFoto.shape[0])
    digitNum = int(settings["digitNum"])
    length = abs(x0 - x1) // digitNum
    model = load_model(modelPath)

    while True:
        foto = GetFoto(cam, channels)
        if showOriginal:
            im = cv2.rectangle(foto, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.imshow("Original", im)
        print('Original shape: ', foto.shape)

        # Crop and resize
        crop = []
        for i in range(0, digitNum):
            temp = foto[y0:y1,
                        x0 + i*length:x0 + (i+1)*length].copy()
            print(temp.shape)
            temp = cv2.resize(temp, (32, 48))
            if channels == 1:
                temp = temp[..., np.newaxis]
            crop.append(temp)

        # Predict
        predict = []
        for im in crop:
            temp = model.predict(np.expand_dims(im, axis=0))
            temp = np.argmax(temp, axis=1)
            predict.append(temp)
        print(predict)
        for number in predict:
            print(number[0], end=' ')
        print()
        if showCrop:
            for i in range(0, digitNum):
                cv2.imshow(str(i), crop[i])
        k = cv2.waitKey(500)
        if k == ord('q'):
            break

if __name__ == "__main__":
    main()
