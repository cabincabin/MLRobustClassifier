import tensorflow as tf
import json
import io
import os
import subprocess
from tensorflow.python.lib.io import file_io
from PIL import Image
from PIL import ImageFilter

imageInputPath = "ImageFiles/"
imageOutputScaled = "Imagefiles256x256/"
imageOutputEdge = "ImagefilesEdge256x256/"
pointsOutput = "PointAnnotationsSet256x256.txt"

def GCPPath(UseGCP):
    if UseGCP == True:
        subprocess.call(["gsutil", "cp", "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet.txt", "PointAnnotationsSet.txt"])
        global imageInputPath, imageOutputScaled, imageOutputEdge, pointsOutput
        imageInputPath = "gs://wpiopenimageskaggle/ImageFiles/"
        imageOutputScaled = "gs://wpiopenimageskaggle/Imagefiles28/"
        imageOutputEdge = "gs://wpiopenimageskaggle/ImagefilesEdge28/"
        pointsOutput = "gs://wpiopenimageskaggle/Imageids/PointAnnotationsSet28.txt"
    getImageGCPPaths()


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
    points = read_file_JSON("PointAnnotationsSet.txt")
    badIndexes = []
    for index, point in enumerate(points):
        uriInp = imageInputPath + point['id'] + ".jpg"
        uricrop = imageOutputScaled + point['id'] + ".jpg"
        uricropEdge = imageOutputEdge + point['id'] + ".jpg"
        #label.append(point['label'])

        size = 256,256
        crop = 0,0,256,256
        if index%100 == 0:
        	print(index)
        try:

            with _open_file_read_binary(uriInp) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('L')
                img = img.resize(size, Image.BICUBIC)
                #imgWEdges = img.copy().filter(ImageFilter.FIND_EDGES)
                with file_io.FileIO(uricrop, mode = 'w') as fThumb:
                    img.save(fThumb, "JPEG")
                #with file_io.FileIO(uricropEdge, mode = 'w') as fEdge:
                    #imgWEdges.save(fEdge, "JPEG")

        except:
            badIndexes.append(index)
    for index in sorted(badIndexes, reverse=True):
        points.pop(index)

    with file_io.FileIO(pointsOutput, mode='w') as pointSaveJson:
        pointSaveJson.write(json.dumps(points))






