import tensorflow.keras.preprocessing.image
from matplotlib import pyplot as plt
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
    img = np.array(img, dtype=np.float)
    scale = 4 if ksize == 1 else pow(2, 2 * ksize - 3)

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = (np.sqrt(np.power(gx, 2) + np.power(gy, 2)) / scale).astype(np.uint8)

    return sobel


def Laplacian(img, ksize = 1):
    img = np.array(img, dtype=np.float)
    scale = 8 if ksize == 1 else pow(2, 2 * ksize - 4)

    laplacian = np.abs(cv2.Laplacian(img, cv2.CV_64F, ksize=ksize, scale = 1 / scale)).astype(np.uint8)

    return laplacian


def Scharr(img):
    scale = 26
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr = (np.sqrt(np.power(gx, 2) + np.power(gy, 2)) / scale).astype(np.uint8)

    return scharr


def Prewitt(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    scale = 6

    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    prewitt = (np.sqrt(np.power(gx, 2) + np.power(gy, 2)) / scale).astype(np.uint8)

    return prewitt


def Roberts(img):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0],])
    scale = 2

    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    roberts = (np.sqrt(np.power(gx, 2) + np.power(gy, 2)) / scale).astype(np.uint8)

    return roberts


def Canny(img, threshold1 = 0, threshold2 = 63):
    canny = cv2.Canny(img, threshold1, threshold2)

    return canny


def PreprocDoG(img):
    img = cv2.bilateralFilter(img, 3, 10, 10)
    img = Sharpen(img, 7, 2)
    img = cv2.bilateralFilter(img, 3, 3, 3)
    img = Sharpen(img, 2, 1)
    RemoveRed(img)
    img = DoG(img, 1, 6)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = CLAHE(img)
    img = ((img / img.max()) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def PreprocDoGThreshold(img):
    img = cv2.bilateralFilter(img, 3, 10, 10)
    img = Sharpen(img, 7, 2)
    img = cv2.bilateralFilter(img, 3, 3, 3)
    img = Sharpen(img, 2, 1)
    RemoveRed(img)
    img = DoG(img, 1, 6)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU, img)
    img = HistogramEqualization(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def PreprocSobel(img):
    img = cv2.bilateralFilter(img, 3, 10, 10)
    img = Sharpen(img, 7, 2)
    img = cv2.bilateralFilter(img, 3, 3, 3)
    img = Sharpen(img, 2, 1)
    RemoveRed(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = Sobel(img, 3)
    img = CLAHE(img)
    img = ((img / img.max()) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def PreprocSobelThreshold(img):
    img = cv2.bilateralFilter(img, 3, 10, 10)
    img = Sharpen(img, 7, 2)
    img = cv2.bilateralFilter(img, 3, 3, 3)
    img = Sharpen(img, 2, 1)
    RemoveRed(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = Sobel(img, 3)
    cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU, img)
    img = HistogramEqualization(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
    map = pool.map(func, dataSplit)
    newData = None if map[0] is None else pd.concat(map)
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
        rotation_range=30,
        width_shift_range=0.,
        height_shift_range=0.,
        #rescale=1. / 255,
        shear_range=20,
        zoom_range=0.2,
        brightness_range=(0.7, 1.3),
        fill_mode='nearest')

    csvFile = pd.read_csv(csvPath + CSVFileName, index_col=0)
    csvFile.dropna()
    workFunc = partial(BatchAugmentation, loadPicturesPath, savePath, readFormat, countAugmentations, indexes, datagen)
    csvFile = ParallelizeData(csvFile, workFunc)
    csvFile.to_csv(csvPath + CSVFileName[:-4] + '_Augmented.csv')


def BatchPreprocessing(folder, src, dst, preprocFunc, names):
    """ The function for preprocessing a batch of images on a single CPU core. """
    for name in names:
        img = cv2.imread(src + folder + name, 1)
        img = cv2.resize(img, (32, 48))
        img = preprocFunc(img)
        cv2.imwrite(dst + folder + name, img)


def DataSetsPreprocessing(folders, src, dst, preprocFunc = PreprocDoG):
    """ Preprocesses data sets using multiprocessing. """
    for folder in folders:
        names = GetAllNames(src + folder)
        workFunc = partial(BatchPreprocessing, folder, src, dst, preprocFunc)
        ParallelizeData(names, workFunc)


def CompareEdgeDetection(folder = "../Images/Original/NewBalanced/", ksize = 11):
    names = GetAllNames(folder)
    for name in names:
        img = cv2.imread("{}{}".format(folder, name), 0)

        dog = DoG(img, 1, 6)
        preproc = PreprocDoGThreshold(img)
        sobel = Sobel(img, ksize)
        prewitt = Prewitt(img)
        canny = Canny(img)
        roberts = Roberts(img)
        scharr = Scharr(img)
        laplacian = Laplacian(img, ksize)

        plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 2), plt.imshow(dog, cmap='gray')
        plt.title('DoG'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 3), plt.imshow(preproc, cmap='gray')
        plt.title('Preproc'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 4), plt.imshow(roberts, cmap='gray')
        plt.title('Roberts'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 5), plt.imshow(prewitt, cmap='gray')
        plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 6), plt.imshow(canny, cmap='gray')
        plt.title('Canny'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 7), plt.imshow(sobel, cmap='gray')
        plt.title('Sobel'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 8), plt.imshow(scharr, cmap='gray')
        plt.title('Scharr'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 9), plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

        plt.show()


def ComparePreprocessing(folder = "../Images/Original/NewBalanced/"):
    names = GetAllNames(folder)
    for name in names:
        img = cv2.imread("{}{}".format(folder, name), 0)
        pD = PreprocDoG(img)
        pS = PreprocSobel(img)
        pDT = PreprocDoGThreshold(img)
        pST = PreprocSobelThreshold(img)

        plt.subplot(3, 3, 1), plt.imshow(pD, cmap='gray')
        plt.title('DoG'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 3), plt.imshow(pS, cmap='gray')
        plt.title('Sobel'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 5), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 7), plt.imshow(pDT, cmap='gray')
        plt.title('DoG + Threshold'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, 9), plt.imshow(pST, cmap='gray')
        plt.title('Sobel + Threshold'), plt.xticks([]), plt.yticks([])

        plt.show()


def CreateExperimentDS():
    datasets = ["Clean/", "NewBalanced/", "Meter_1/", "Meter_2/", "YouTube/"]
    dstFolders = ["DoGThreshold/", "DoG/", "SobelThreshold/", "Sobel/"]
    src = "../Images/Augmented/"
    dst = "../Images/Preprocessed/"
    preprocFuncs = [PreprocDoGThreshold, PreprocDoG, PreprocSobelThreshold, PreprocSobel]
    for idx, preprocFunc in enumerate(preprocFuncs):
        DataSetsPreprocessing(datasets, src, dst + dstFolders[idx], preprocFunc)



def Main():
    CreateExperimentDS()

if __name__ == "__main__":
    Main()