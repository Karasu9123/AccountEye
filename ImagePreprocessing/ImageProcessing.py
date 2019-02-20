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
    new_csv_file.to_csv(savePath + '/' + CSVFileName[:-4] + '+Augmentation.csv')
    cv2.waitKey(0)

'''
def HOG():
    hog = cv2.HOGDescriptor(_winSize=(32,48), _blockSize=(4,4), _blockStride=(2,2), _cellSize=(2,2), _nbins=9, _gammaCorrection=True)
    magnitudes, angles = hog.computeGradient(image)
    sobel = cv2.imshow("digit %s" % i, np.sqrt(np.power(magnitudes[..., 0],2) + np.power(magnitudes[..., 1],2)))
    return sobel 
'''

def Sobel(img, ksize=1):
    img = np.float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.uint8(np.sqrt(np.power(gx, 2) + np.power(gy, 2)))

def CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2,2))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

def main():
    NewAugmentation("/home/Shared/AccountEye/Images/Labels",
                    "Cam_2 (copy).csv",
                    "/home/Shared/AccountEye/Images/Resized/Meter_2",
                    "/home/Shared/AccountEye/Images/Augmented/Meter_2")

if __name__ == "__main__":
    main()

