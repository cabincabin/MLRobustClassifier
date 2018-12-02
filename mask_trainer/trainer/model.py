import tensorflow as tf
import json
import io
import os
import subprocess
from tensorflow.python.lib.io import file_io
from PIL import Image
from PIL import ImageFilter
import cv2

imageInputPath = "ImageFiles/"
imageOutputScaled = "Imagefiles256x256/"
imageOutputEdge = "ImagefilesEdge256x256/"
pointsOutput = "PointAnnotationsSet256x256.txt"

def GCPPath(UseGCP):
    if UseGCP == True:
        getImageGCPPaths()
        """subprocess.call(["gsutil", "cp", "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet.txt", "PointAnnotationsSet.txt"])
        global imageInputPath, imageOutputScaled, imageOutputEdge, pointsOutput
        imageInputPath = "gs://wpiopenimageskaggle/ImageFiles/"
        imageOutputScaled = "gs://wpiopenimageskaggle/Imagefiles256x256/"
        imageOutputEdge = "gs://wpiopenimageskaggle/ImagefilesEdge256x256/"
        pointsOutput = "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet256x256.txt"
        """

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


def getImageGCPPaths():
    global imageInputPath, imageOutputScaled, imageOutputEdge, pointsOutput
    #points = read_file_JSON("PointAnnotationsSet.txt")
    points = read_file_JSON("/home/saina/School/Machine Learning/MLRobustClassifier/mask_trainer/trainer/PointAnnotationsSet.txt")
    badIndexes = []
    for index, point in enumerate(points):
        uriInp = imageInputPath + point['id'] + ".jpg"
        uricrop = imageOutputScaled + point['id'] + ".jpg"
        uricropEdge = imageOutputEdge + point['id'] + ".jpg"
        #label.append(point['label'])

        size = 256,256
        crop = 0,0,256,256
        try:

            with _open_file_read_binary(uriInp) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('L')

                cv2.imshow('image',img)
                cv2.waitKey(0)

                x0 = point['annotation'][0]['x0']
                x1 = point['annotation'][0]['x1']
                y0 = point['annotation'][0]['y1']
                y1 = point['annotation'][0]['y1']
                print x0, x1, y0, y1

                sum = 0
                for j in range(257):
                    for i in range(257):
                        sum += img[j][i]
                average = sum/(256*256)

                for j in range(257):
                    for i in range(257):
                        if (i < x0 or i > x1) and (j < y0 or j > y1):
                            if average >= 127:
                                img[j][i] = 255
                            else:
                                img[j][i] = 0

                img = img.resize(size, Image.ANTIALIAS)
                imgWEdges = img.copy().filter(ImageFilter.FIND_EDGES)
                with file_io.FileIO(uricrop, mode = 'wb') as fThumb:
                    img.save(fThumb, "JPEG")
                with file_io.FileIO(uricropEdge, mode = 'wb') as fEdge:
                    imgWEdges.save(fEdge, "JPEG")

        except:
            badIndexes.append(index)
    for index in sorted(badIndexes, reverse=True):
        points.pop(index)

    with file_io.FileIO(pointsOutput, mode='w') as pointSaveJson:
        pointSaveJson.write(json.dumps(points))






