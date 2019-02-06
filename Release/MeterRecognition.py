import cv2
import numpy as np

def DenoiseSobel(im):
    im = cv2.fastNlMeansDenoising(im, h=4)
    im = CLAHE(im)
    im = Sobel(im)

def CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2,2))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def Sobel(img, _ksize = 1):
    img = np.float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=_ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=_ksize)
    return np.uint8(np.sqrt(np.power(gx, 2) + np.power(gy, 2)))

def GetFoto():
    #cam = cv2.VideoCapture(0)
    #ret, frame = cam.read()
    foto = cv2.imread("meter.jpg", 1)
    #foto = Sobel(foto)
    cv2.imshow("original", foto)
    print(foto.shape)
    return foto

def main():
    import json
    import Experiments.Models as M

    with open('settings.json') as f:
        settings = json.load(f)
    x0 = int(settings["region"]["x0"])
    x1 = int(settings["region"]["x1"])
    y0 = int(settings["region"]["y0"])
    y1 = int(settings["region"]["y1"])
    digitNum = int(settings["digitNum"])
    length = abs(x0-x1)//digitNum

    foto = GetFoto()

    crop = []
    for i in range(0, digitNum):
        temp = foto[y0:y1,
                    x0 + i*length:x0 + (i+1)*length].copy()
        print(temp.shape)
        temp = cv2.resize(temp, (32, 48))#[..., np.newaxis]
        crop.append(temp)

    model = M.LoadModel("/home/Shared/AccountEye/Experiments/log/ResNet_All-130-0.89.hdf5")
    predict = []
    for im in crop:
        temp = model.predict(np.expand_dims(im, axis=0))
        temp = np.argmax(temp, axis=1)
        predict.append(temp/2)

    print(predict)
    print('43693009')
    #for i in range(0, digitNum):
    #    cv2.imshow(str(i), crop[i])
    cv2.waitKey(0)



if __name__ == "__main__":
    main()

# TODO: прочитать парамметры
# TODO: сделать фотку
# TODO: вырезать
# TODO: распознать
