import random
import subprocess
from tensorflow.python.lib.io import file_io
from PIL import Image
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
        ftrArry = np.asarray(ftr, dtype="int32")

    ProcessArraysCarlo(imgArray, ftrArry)


def ProcessArraysCarlo(imgArray, ftrArry):
    PointsToStart = [[random.randint(0, 256), random.randint(0, 256), random.randint(0, 359)] for x in range(50)]
    ftr256 = np.multiply(np.ones((256,256),dtype=np.int),-1)
    x,y = 0,0
    ftr256[x:ftrArry.shape[0], y:ftrArry.shape[1]] = ftrArry

    print(ftr256)

if __name__== "__main__":

    processAsImagesFromFiles("00a9f6e69fd697b3.jpg","002200be72145198.jpg", True)

