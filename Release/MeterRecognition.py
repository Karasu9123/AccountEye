import cv2
import numpy as np
import ImagePreprocessing.ImageProcessing as IP
from tensorflow.keras.models import load_model


def GetCam(idCam = 0):
    cam = cv2.VideoCapture(idCam)
    cam.set(cv2.CAP_PROP_EXPOSURE, -5)
    return cam

def GetPhoto(cam=None, path=''):
    if cam is None:
        photo = cv2.imread(path, 1)
    else:
        ret, photo = cam.read()
        photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)

    return photo

def LoadSettings(settingsPath):
    import json
    with open(settingsPath) as f:
        settings = json.load(f)
        x0 = float(settings["region"]["x0"])
        x1 = float(settings["region"]["x1"])
        y0 = float(settings["region"]["y0"])
        y1 = float(settings["region"]["y1"])
        digitNum = int(settings["digitNum"])
    return x0, y0, x1, y1, digitNum

def CropAndPreproc(photo, x0, y0, x1, y1, digitNum, channels):
    crop = []
    areaLength = abs(x0 - x1) // digitNum
    for i in range(0, digitNum):
        left = x0 + i * areaLength
        right = left + areaLength
        digit_i = photo[y0:y1, left:right].copy()
        resized = cv2.resize(digit_i, (32, 48))
        preprocessed = IP.PreprocDoG(resized)
        if channels == 1:
            preprocessed = preprocessed[..., np.newaxis]
        elif channels == 3:
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        crop.append(preprocessed)
    return crop

def Predict(crop, model):
    predict = []
    for im in crop:
        temp = model.predict(np.expand_dims(im, axis=0))
        temp = np.argmax(temp, axis=1)
        predict.append(temp[0])
    return predict

def Main():
    fromfiles = True
    onlypreproc = False
    imdir = "Test/"
    settdir = "Settings/"
    modelPath = 'ResNet_All_Blur-09-0.99.hdf5'
    settpath = "Settings/settings.json"
    impath = ""
    showOriginal = True
    showCrop = False
    channels = 1
    width, height = 1024, 682
    pause = 0
    i = 0
    imnames = None

    model = load_model(modelPath)
    model = model.layers[-2]

    if fromfiles:
        cam = None
        imnames = IP.GetAllNames(imdir)
        print("images: ", imnames)
    else:
        cam = GetCam()
        pause = 1

    while True:
        if fromfiles:
            if i >= len(imnames):
                break
            settname = imnames[i].replace(".jpg", "") + ".json"
            settpath = settdir + settname
            impath = imdir + imnames[i]
            i += 1

        x0, y0, x1, y1, digitNum = LoadSettings(settpath)
        x0 = int(x0 * width)
        x1 = int(x1 * width)
        y0 = int(y0 * height)
        y1 = int(y1 * height)

        photo = GetPhoto(cam, impath)
        photo = cv2.resize(photo, (width, height))

        if onlypreproc:
            photo = IP.PreprocDoG(photo)
        else:
            crop = CropAndPreproc(photo, x0, y0, x1, y1, digitNum, channels)
            predict = Predict(crop, model)

            # Show predict.
            for j in range(digitNum):
                print(predict[j] / 2, end = "  ")
            print()

            if showCrop:
                for j in range(0, digitNum):
                    cv2.imshow(str(j), crop[j])

        if showOriginal:
            im = cv2.rectangle(photo, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.imshow("Original", im)


        k = cv2.waitKey(pause)
        if k == ord('q'):
            break


if __name__ == "__main__":
    Main()
