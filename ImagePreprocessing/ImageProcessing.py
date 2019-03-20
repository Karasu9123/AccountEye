from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from functools import partial
import numpy as np
import cv2
import os
import pandas as pd
import multiprocessing as mp
import shutil as sh

CORE_COUNT = 12

def GetAllNames(path):
    return os.listdir(path)

def parallelize_data(data, func):
    data_split = np.array_split(data, CORE_COUNT)
    pool = mp.Pool(processes=CORE_COUNT)
    new_data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return  new_data

def MakeAugmentation(datagen, loadPicturesPath, readFormat, countAugmentations, savePath, indexes, csv_file):
    element_amount = len(csv_file)
    for i in range(element_amount):
        file = csv_file.iloc[i].name
        sh.copyfile(loadPicturesPath + '/' + file, savePath + '/' + file)
        image = cv2.imread(loadPicturesPath + '/' + file, readFormat)[np.newaxis, ...]
        aug_counter = 0
        for batch in datagen.flow(image, batch_size=1):
            if aug_counter == countAugmentations:
                break
            aug_name = file[:-4] + '_A{}.jpg'.format(aug_counter)
            cv2.imwrite(savePath + '/' + aug_name, batch[0, ...])
            row = pd.Series([csv_file.iloc[i][0], csv_file.iloc[i][1], csv_file.iloc[i][2]], index=indexes)
            row.name = aug_name
            csv_file = csv_file.append(row)
            aug_counter += 1
    return  csv_file

def Rename(path, newName):
    i = 0
    l = GetAllNames(path)
    for file in l:
        newname = newName + '_' + str(i) + ".jpg"
        src = path + file
        newname = path + newname
        os.rename(src, newname)
        i += 1

def Resize(loadPath, savePath, outputSize = (32, 48), readFormat = 1):
    names = GetAllNames(loadPath)
    for name in names:
        image = cv2.imread(loadPath+'/'+name, readFormat)
        image = cv2.resize(image, outputSize)
        cv2.imwrite(savePath+'/'+name, image)

def NewAugmentation(loadCSVFilePath, CSVFileName, loadPicturesPath, savePath, readFormat = 1, countAugmentations = 500):

    columns = ['Image_Name', '10_Classes', '20_Classes_Float', '20_Classes']
    indexes = ['10_Classes', '20_Classes_Float', '20_Classes']
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.1, 1.2),
        fill_mode='nearest')

    csv_file = pd.read_csv(loadCSVFilePath + '/' + CSVFileName, index_col=0)
    csv_file.dropna()
    func = partial(MakeAugmentation, datagen, loadPicturesPath, readFormat, countAugmentations, savePath, indexes)
    csv_file = parallelize_data(csv_file, func)
    csv_file.to_csv(savePath + '/' + CSVFileName[:-4] + '_Blur+Augmentation.csv')
    cv2.waitKey(0)

def hog(im, ksize = 1):
    im = np.float32(im)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=ksize)
    grad = np.uint8(np.sqrt(np.power(gx, 2) + np.power(gy, 2)))
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return grad

def sobel(img, ksize=1):
    img = np.float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.uint8(np.sqrt(np.power(gx, 2) + np.power(gy, 2)))

def clahe(img, clipLimit=5.0, tileGridSize=(3, 3)):
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

def dog(im, inner, outer):
    innerBlur = cv2.GaussianBlur(im, (0, 0), inner).astype(np.int)
    outerBlur = cv2.GaussianBlur(im, (0, 0), outer).astype(np.int)
    return np.clip(innerBlur - outerBlur, 0, 255).astype(np.ubyte)

def equalize(im):
    if im.ndim == 3:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        im = cv2.equalizeHist(im)
    return im

def sharpen(im, sigma, amount):
    blur = cv2.GaussianBlur(im, (0, 0), sigma)
    im = (amount + 1) *  im.astype(np.int) - amount * blur.astype(np.int)
    return np.clip(im, 0, 255).astype(np.ubyte)

def preproc(im):
    im = cv2.bilateralFilter(im, 3, 10, 10)
    im = sharpen(im, 7, 2)
    im = cv2.bilateralFilter(im, 3, 3, 3)
    im = sharpen(im, 2, 1)
    mask = (im[:, :, 2] > cv2.add(im[:, :, 0], im[:, :, 1]))
    im[mask] = 0
    im[:, :, 2] = 0
    im = dog(im, 1, 6)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = equalize(im)
    return im

def MakePreproc(s, names):
    for name in names:
        im = cv2.imread("../Images/Augmented/" + s + name, 1)
        im = cv2.resize(im, (32, 48))
        im = preproc(im)
        cv2.imwrite("../Images/Preproc/" + s + name, im)

def main():
    """NewAugmentation("../Images/Labels",
                    "YouTube.csv",
                    "../Images/Original",
                    "../Images/Augmented", countAugmentations=500)
    """
    NewAugmentation("../Images/Labels",
                    "Clean.csv",
                    "../Images/Original",
                    "../Images/Augmented", countAugmentations=1000)

    sets = ["Clean/"]
    for s in sets:
        names = GetAllNames("../Images/Augmented/" + s)
        names = np.array_split(names, CORE_COUNT)
        pool = mp.Pool(processes=CORE_COUNT)
        func = partial(MakePreproc, s)
        pool.map(func, names)
        pool.close()
        pool.join()
    """
    im = cv2.imread("/opt/AccountEye/Images/Original/NewBalanced/73.png",  1)
    im = cv2.resize(im, (32, 48))
    im = preproc(im)
    cv2.waitKey(0)"""


if __name__ == "__main__":
    main()