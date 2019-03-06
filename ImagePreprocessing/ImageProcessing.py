from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import cv2
import os
import pandas as pd

def GetAllNames(path):
    return os.listdir(path)

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

def Augmentation(loadPath, savePath, readFormat = 1, countAugmentations = 20):

    # TODO: GetAllNames change
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.01, 1.2),
        fill_mode='nearest')

    names = GetAllNames(loadPath)

    for name in names:
        image = cv2.imread(loadPath+'/'+name, readFormat)[np.newaxis, ...]
        i = 0
        # TODO: implement saving augmented image
        # https://keras.io/preprocessing/image/
        for batch in datagen.flow(image, batch_size=1):
            i += 1
            if i > countAugmentations:
                break

def NewAugmentation(loadCSVFilePath, CSVFileName, loadPicturesPath, savePath, readFormat = 1, countAugmentations = 1):

    columns = ['Image_Name', '10_Classes', '20_Classes_Float', '20_Classes']
    indexes = ['10_Classes', '20_Classes_Float', '20_Classes']
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.01, 1.2),
        fill_mode='nearest')

    csv_file = pd.read_csv(loadCSVFilePath + '/' + CSVFileName, index_col=0)
    new_csv_file = csv_file.copy()
    new_csv_file.dropna()
    element_amount = len(new_csv_file)
    for i in range(element_amount):
        file = csv_file.iloc[i].name
        image = cv2.imread(loadPicturesPath + '/' + file, readFormat)[np.newaxis, ...]
        aug_counter = 0
        for batch in datagen.flow(image, batch_size=1):
            if aug_counter == countAugmentations:
                break
            aug_name = file[:-4] + '_A{}.jpg'.format(aug_counter)
            cv2.imwrite(savePath + '/' + aug_name, batch[0, ...])
            row = pd.Series([new_csv_file.iloc[i][0], new_csv_file.iloc[i][1], new_csv_file.iloc[i][2]], index=indexes)
            row.name = aug_name
            new_csv_file = new_csv_file.append(row)
            aug_counter += 1
    new_csv_file.to_csv(savePath + '/' + CSVFileName[:-4] + '_Blur+Augmentation.csv')
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
    cv2.erode(im, (3, 3), im, iterations=1)
    im = cv2.bilateralFilter(im, 3, 10, 10)
    im = sharpen(im, 7, 2)
    im = cv2.bilateralFilter(im, 3, 3, 3)
    im = sharpen(im, 2, 1)
    mask = (im[:, :, 2] > cv2.add(im[:, :, 0], im[:, :, 1]))
    im[mask] = 0
    im[:, :, 2] = 0
    im = dog(im, 1, 6)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.threshold(im, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU, im)
    im = sharpen(im, 10, 1)
    im = sobel(im)
    cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, im)
    return im

def main():
    """NewAugmentation("/home/shared/AccountEye/Images/Labels",
                    "All.csv",
                    "/home/shared/AccountEye/Images/BlurAndThreshold",
                    "/home/shared/AccountEye/Images/Augmented/BlurAll")"""
    sets = ["Other/", "YouTube/", "Meter_1/", "Meter_2/"]
    for s in sets:
        names = GetAllNames("../Images/Original/" + s)
        for name in names:
            im = cv2.imread("../Images/Original/" + s + name, 1)
            im = cv2.resize(im, (32, 48))
            im = preproc(im)
            cv2.imwrite("../Images/BlurAndThreshold/" + s + name, im)

if __name__ == "__main__":
    main()