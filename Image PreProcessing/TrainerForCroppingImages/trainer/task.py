
import trainer.model as model
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

global imageInputPath
imageInputPath = "Imagefiles256x256/"

##Use TRUE to deploy to GCP ML engine
def USEGCP(UseGCP):
    if UseGCP == True:
        args = parser.parse_args()
        print(args.points_path)
        print(args.images_input_path)
        subprocess.call(["gsutil", "cp", args.points_path, "PointAnnotationsSet256x256.txt"])
        subprocess.call(["gsutil", "cp", "gs://mlclassifiertuning/imageIds/PointAnnotationsLabel.txt", "PointAnnotationsLabel.txt"])
        global imageInputPath, imagePoint, outputPath
        imagePoint = "PointAnnotationsSet256x256.txt"
        imageInputPath = args.images_input_path#"gs://wpiopenimageskaggle/Imagefiles256x256/"
        outputPath = "gs://wpiopenimageskaggle/imageIds/"

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
def CreateBatchOfImages(batchSize, labelDict, imagePath):
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
                    uriInp = imagePath + imageId + ".jpg"

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
#        for labelkey in labelDict.keys(): #This could be expensive, may want to find better way
#            if len(labelDict[labelkey])==0:
#                return ImageLabelDict, labelDict, SuccessNum #sucsess num could have offby1error keep an eye out
#        SuccessNum += 1
    return ImageLabelDict, labelDict, SuccessNum

def getIDLabelDict(points):
    idDict = {}
    for point in points:
        idDict.setdefault(point['id'], point['annotations'][0]['label'])
    return idDict

#for local testing
#imageInputPath = "Imagefiles256x256/"
#imagePoint = "PointAnnotationsSet256x256.txt"
#outputPath
#this will need to be  moved to 'task.py' when all code is done, in order to deploy to GCP ML engine
# Shoop testing
parser = argparse.ArgumentParser()
parser.add_argument('--points-path', dest='points_path', required=False)
#FOR CROPPED BLACK AND WHITE IMAGES USE:
    #---->gs://wpiopenimageskaggle/Imagefiles256x256/
#FOR CROPPED EDGE IMAGES USE:
    #---->gs://wpiopenimageskaggle/ImagefilesEdge256x256/
parser.add_argument('--images-input-path', dest='images_input_path', required=False)
#parser.add_argument('--images-output-path', dest='images_output_path', required=False)

#TO DEPLOY IN GCP ML ENGINE, MUST DELETE ALL LOCAL IMAGE FOLDERS AND 'PointAnnotationsSet"
#TO RUN
#IN GOOGLE CLOUD CONSOLE
#1. >>>REGION=us-central1
#2. >>>JOB_NAME=<GIVEITANAMEANDVERSION>
#3. >>> gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://mlengine_example_bucket --runtime-version 1.8 --module-name trainer.task --package-path trainer/ --region $REGION
if __name__== "__main__":
    global imageInputPath
    #USEGCP(True) #SET TO TRUE WHEN USING GCP
    pointsTest = read_file_JSON("PointAnnotationsLabel.txt") #create points from JSON: IDS are key, contains label/confidince/crop.
    points = read_file_JSON("PointAnnotationsSet256x256.txt") #create points from JSON: IDS are key, contains label/confidince/crop.
    IdsFromLabels = CreateDictLabels(points) #creates a dictionary such that the key is a label, returns all IDS of that label
    #the batch function removes ids from the dictionary, so keep an eye on that.
    idsToRemoveFromEachBatch = IdsFromLabels.copy()
    # Creates a batch in ImageLabelDict in the format [ImageArr, ID]
    ImageLabelDict, idsToRemoveFromEachBatch, SuccessNum = CreateBatchOfImages(500, idsToRemoveFromEachBatch,imageInputPath)
    #ImageLabel = {1: [[0],'9eb39d618fd92994'], 2: [[0],'025a6cccec41134'], 3: [[0],'002200be72145198'], 4:[[0],'003d4acb635f05b7']}

    
    IdsFromLabelsTest = CreateDictLabels(pointsTest) #creates a dictionary such that the key is a label, returns all IDS of that label
    #the batch function removes ids from the dictionary, so keep an eye on that.
    idsToRemoveFromEachBatchTest = IdsFromLabelsTest.copy()
    # Creates a batch in ImageLabelDict in the format [ImageArr, ID]
    testDict, idsToRemoveFromEachBatchTest, SuccessNumTest = CreateBatchOfImages(250, idsToRemoveFromEachBatchTest, "Imagefiles256x256/")
    #ImageLabel = {1: [[0],'9eb39d618fd92994'], 2: [[0],'025a6cccec41134'], 3: [[0],'002200be72145198'], 4:[[0],'003d4acb635f05b7']}

    #model.main(ImageLabel, getIDLabelDict(points), SuccessNum, "")
    #"gs://wpiopenimageskaggle/imageIds/edgeminiResults/"
    #"gs://wpiopenimageskaggle/imageIds/256miniResults/"
    model.main(ImageLabelDict, getIDLabelDict(points), SuccessNum, "gs://mlclassifiertuning/imageIds/edgeminiResultsLabel/", testDict)

