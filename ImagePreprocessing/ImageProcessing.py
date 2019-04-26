import tensorflow.keras.preprocessing.image
from functools import partial
import numpy as np
import cv2
import os
import pandas as pd
import multiprocessing as mp
import shutil as sh

CPU_CORES = 10



def GetAllNames(path):
    return os.listdir(path)


def RenameAll(path, newName):
    names = GetAllNames(path)
    for i, name in enumerate(names):
        newName = newName + '_' + str(i) + ".jpg"
        src = path + name
        dst = path + newName
        os.rename(src, dst)


def ResizeAll(loadPath, savePath, outputSize = (32, 48), readFormat = 1):
    names = GetAllNames(loadPath)
    for name in names:
        img = cv2.imread(loadPath + name, readFormat)
        img = cv2.resize(img, outputSize)
        cv2.imwrite(savePath + name, img)


def HistogramEqualization(img):
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        img = cv2.equalizeHist(img)

    return img


def CLAHE(img, clipLimit=5.0, tileGridSize=(3, 3)):
    """ Contrast Limited Adaptive Histogram Equalization. """
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        img = clahe.apply(img)

    return img


def Sharpen(img, sigma, amount):
    """ The Sharpen filter enhance colors and accentuates edges but also any noise or blemish
     and it may create noise in graduated color areas like the sky or a water surface. """
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, amount + 1, blur, -amount, 0)

    return sharpened


def DoG(img, inner, outer):
    """ Difference of Gaussians is a feature enhancement algorithm. """
    innerBlur = cv2.GaussianBlur(img, (0, 0), inner)
    outerBlur = cv2.GaussianBlur(img, (0, 0), outer)
    diff = cv2.addWeighted(innerBlur, 1, outerBlur, -1, 0)

    return diff


def RemoveRed(img):
    """ Removes red pixels from the image. """
    if img.ndim == 3:
        mask = (img[:, :, 2] > cv2.add(img[:, :, 0], img[:, :, 1]))
        img[mask] = 0
        img[:, :, 2] = 0


def Sobel(img, ksize = 1):
    img = np.float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)

    return np.uint8(np.sqrt(np.power(gx, 2) + np.power(gy, 2)))


def Preprocessing(img):
    """ Preprocesses the color image and returns it in grayscale. """
    img = cv2.bilateralFilter(img, 3, 10, 10)
    img = Sharpen(img, 7, 2)
    img = cv2.bilateralFilter(img, 3, 3, 3)
    img = Sharpen(img, 2, 1)
    RemoveRed(img)
    img = DoG(img, 1, 6)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = HistogramEqualization(img)

    return img


def HoG(img, shape = (32, 48)):
    """ Histogram of Oriented Gradients. """
    winSize = shape
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hist = hog.compute(img)

    return hist


def ParallelizeData(data, func):
    """ Splits workflow (func) on data between multiple CPU cores. """
    dataSplit = np.array_split(data, CPU_CORES)
    pool = mp.Pool(processes=CPU_CORES)
    newData = pd.concat(pool.map(func, dataSplit))
    pool.close()
    pool.join()

    return newData


def BatchAugmentation(loadPicturesPath, savePath, readFormat, countAugmentations, indexes, datagen, csvFile):
    """ The function for augmentation a batch of images on a single CPU core. """
    elementAmount = len(csvFile)
    for i in range(elementAmount):
        file = csvFile.iloc[i].name
        sh.copyfile(loadPicturesPath + file, savePath + file)
        img = cv2.imread(loadPicturesPath + file, readFormat)[np.newaxis, ...]
        augCounter = 0
        for batch in datagen.flow(img, batch_size=1):
            if augCounter == countAugmentations:
                break
            augName = file[:-4] + '_A{}.jpg'.format(augCounter)
            cv2.imwrite(savePath + augName, batch[0, ...])
            row = pd.Series([csvFile.iloc[i][0], csvFile.iloc[i][1], csvFile.iloc[i][2]], index=indexes)
            row.name = augName
            csvFile = csvFile.append(row)
            augCounter += 1

    return csvFile


def Augmentation(csvPath, CSVFileName, loadPicturesPath, savePath, readFormat = 1, countAugmentations = 500):
    """ Augments single dataset using multiprocessing. """

    indexes = ['10_Classes', '20_Classes_Float', '20_Classes']
    datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.1, 1.2),
        fill_mode='nearest')

    csvFile = pd.read_csv(csvPath + CSVFileName, index_col=0)
    csvFile.dropna()
    workFunc = partial(BatchAugmentation, loadPicturesPath, savePath, readFormat, countAugmentations, indexes, datagen)
    csvFile = ParallelizeData(csvFile, workFunc)
    csvFile.to_csv(csvPath + CSVFileName[:-4] + '_Augmented.csv')


def BatchPreprocessing(s, names):
    """ The function for preprocessing a batch of images on a single CPU core. """
    for name in names:
        img = cv2.imread("../Images/Augmented/" + s + name, 1)
        img = cv2.resize(img, (32, 48))
        img = Preprocessing(img)
        cv2.imwrite("../Images/Preproc/" + s + name, img)


def DataSetsPreprocessing(sets):
    """ Preprocesses data sets using multiprocessing. """
    for s in sets:
        names = GetAllNames("../Images/Augmented/" + s)
        workFunc = partial(BatchPreprocessing, s)
        ParallelizeData(names, workFunc)


def Main():
    sets = ["NewBalanced/"]
    """Augmentation("../Images/Labels/",
                    "NewBalanced.csv",
                    "../Images/Original/",
                    "../Images/Augmented/", countAugmentations=100)"""

    DataSetsPreprocessing(sets)



if __name__ == "__main__":
    Main()
