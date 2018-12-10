import random
import subprocess

import scipy as scipy
from tensorflow.python.lib.io import file_io
from PIL import Image
from PIL import ImageChops
import numpy as np
import io
from PIL import ImageFilter


def processAsImagesFromFiles(ImageFile, FeatureFile, shouldCrop=False): #ImageFile is 256 by 256 image, featurefile is the feature as an image, must be less than 256.
    #Step 1: populate


    with file_io.FileIO(ImageFile, mode='r') as imgio:
        image_bytes = imgio.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        imgArray = np.asarray(img, dtype="int32")

    with file_io.FileIO(FeatureFile, mode='r') as ftrio:
        image_bytes = ftrio.read()
        if shouldCrop:
            crop = 0, 0, 25, 25
            ftr = (Image.open(io.BytesIO(image_bytes)).convert('L')).crop(crop)
        else:
            ftr = (Image.open(io.BytesIO(image_bytes)).convert('L'))
        ftrArry = np.asarray(ftr)

    ProcessArraysCarlo(img, imgArray, ftrArry)


def ProcessArraysCarlo(img, imgArray, ftrArry):
    PointsToStart = [[random.randint(0, 256), random.randint(0, 256), random.randint(0, 45)] for x in range(50)]
    ftr256 = np.multiply(np.ones((256,256),dtype=np.int),-1)
    x,y = 0,0
    ftr256[x:ftrArry.shape[0], y:ftrArry.shape[1]] = ftrArry
    WeightsPerIndex= []
    for index, point in enumerate(PointsToStart):
        ftrcp = ftr256.copy()
        ftrcp = scipy.ndimage.shift(ftrcp, (point[1], point[2]))
        ftrcp = scipy.ndimage.rotate(ftrcp,point[2],reshape=False)
        weights = []
        for r in range(256):
            for c in range(256):
                if ftrcp[r,c] != -1:
                    weights.append(abs(imgArray[r,c]-ftrcp[r,c]))
                else:
                    ftrcp[r,c] = 0
        if len(weights)!=0:
            WeightsPerIndex.append([ftrcp,(sum(weights)/len(weights))])
    min = WeightsPerIndex[0][1]
    minIndex = 0
    for index, tup in enumerate(WeightsPerIndex):
        if tup[1] < min:
            min = tup[1]
            minIndex = index
    ftrchoice = WeightsPerIndex[minIndex][0]
    featureImage = Image.fromarray(ftrArry)
    featureImage.show()
    #multimage = ImageChops.multiply(featureImage, img)
    #multimage.show()




    print(ftr256)

#if __name__== "__main__":
def main():
    processAsImagesFromFiles("00a9f6e69fd697b3.jpg","002200be72145198.jpg", True)

'''  PIL.ImageChops.multiply(image1, image2)
Superimposes two images on top of each other.

If you multiply an image with a solid black image, the result is black. If you multiply with a solid white image, the image is unaffected.
'''