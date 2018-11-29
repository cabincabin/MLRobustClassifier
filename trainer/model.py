import tensorflow as tf
import json
import io
import os
import subprocess
from tensorflow.python.lib.io import file_io
from PIL import Image
import numpy as np
import argparse
from PIL import ImageFilter

##Use TRUE to deploy to GCP ML engine
def USEGCP(UseGCP):
    if UseGCP == True:
        args = parser.parse_args()
        subprocess.call(["gsutil", "cp", args.points_path, "PointAnnotationsSet.txt"])
        global imageInputPath
        imageInputPath = args.images_input_path

#Dumps a file to JSON format, used to get points from 'PointAnnotationsSet.txt'
def read_file_JSON(filename):

    with open(filename, 'r') as file:
        # self.response.write(cloudstorage_file.read())
        annotString = file.read()
        file.close()

    annotations = json.loads(annotString)
    return annotations

def _open_file_read_binary(uri):
    try:
        return file_io.FileIO(uri, mode='r')
    except:
        return file_io.FileIO(uri, mode='rb')

#given a set of JSON, in form
#[{"id": "9eb39d618fd92994", "annotations": [{"confidence": "1", "label": "/m/01g317", "y1": "0.807014", "y0": "0.218368", "x0": "0.072245", "x1": "0.729939", "id": "9eb39d618fd92994"}]}
#create a dictionary: labels are the key, returns a list of all of the ids in that label
def CreateDictLabels(Points):
    dictOfLabels = {}
    for point in Points:
        labelkey = point['annotations'][0]['label'] #get the label
        if labelkey not in dictOfLabels:
            dictOfLabels.setdefault(labelkey, [])
        dictOfLabels[labelkey].append(point['id'])
    return dictOfLabels

#Given a label dictionary and a batch size, create and return:
#A label dictionary containing a set of tuples [[imageArr,id],[imageArr,id]...]
#the ORIGINAL DICTIONARY with the chosen images removed
#The max number of images that were added per label,
    #---->this is for if a label has less images than another, so that the user
    #---->will not train more images of one label than another, which would cause weighting issues.
    #---->IE label1: 5 images, label2: 10 images, label3: 7 images.
    #---->5 images of each label would be added to the output dictionary and the number 5 would be returned
def CreateBatchOfImages(batchSize, labelDict):
    global imageInputPath
    ImageLabelDict = {}
    SuccessNum = 0
    for index in range(batchSize):
        for labelkey in labelDict.keys():
            if labelkey not in ImageLabelDict:
                ImageLabelDict.setdefault(labelkey, [])

            #force add one image, keep trying until one has been added, protects against corrupted images
            while len(labelDict[labelkey]) != 0:
                try:
                    imageId = labelDict[labelkey].pop(0)
                    uriInp = imageInputPath + imageId + ".jpg"

                    with _open_file_read_binary(uriInp) as f:
                        image_bytes = f.read()
                        img = Image.open(io.BytesIO(image_bytes))
                        imgArr = np.array(img)
                except: #find image exception error for better style
                    continue
                else:
                    ImageLabelDict[labelkey].append([imgArr, imageId])
                    break
        #make sure all have nonzero number of images left
        for labelkey in labelDict.keys(): #This could be expensive, may want to find better way
            if len(labelDict[labelkey])==0:
                return ImageLabelDict, labelDict, SuccessNum #sucsess num could have offby1error keep an eye out
        SuccessNum += 1
    return ImageLabelDict, labelDict, SuccessNum

#for local testing
imageInputPath = "Imagefiles256x256/"
imagePoint = "PointAnnotationsSet.txt"

#this will need to be  moved to 'task.py' when all code is done, in order to deploy to GCP ML engine
parser = argparse.ArgumentParser()
parser.add_argument('--points_path', dest='points_path', required=False)
#FOR CROPPED BLACK AND WHITE IMAGES USE:
    #---->gs://wpiopenimageskaggle/Imagefiles256x256/
#FOR CROPPED EDGE IMAGES USE:
    #---->gs://wpiopenimageskaggle/ImagefilesEdge256x256/
parser.add_argument('--images_input_path', dest='images_input_path', required=False)

def main():
    USEGCP(False) #SET TO TRUE WHEN USING GCP
    points = read_file_JSON(imagePoint) #create points from JSON: IDS are key, contains label/confidince/crop.
    IdsFromLabels = CreateDictLabels(points) #creates a dictionary such that the key is a label, returns all IDS of that label
    #the batch function removes ids from the dictionary, so keep an eye on that.
    idsToRemoveFromEachBatch = IdsFromLabels.copy()
    # Creates a batch in ImageLabelDict in the format [ImageArr, ID]
    ImageLabelDict, idsToRemoveFromEachBatch, SuccessNum = CreateBatchOfImages(10, idsToRemoveFromEachBatch)
    a = ImageLabelDict[0]







