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
        imageOutputScaled = "gs://wpiopenimageskaggle/Imagefilesbbox256/"
        imageOutputEdge = "gs://wpiopenimageskaggle/ImagefilesbboxEdge/"
        pointsOutput = "gs://wpiopenimageskaggle/Imageids/PointAnnotationsSet256bbox.txt"
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

        crop = (round(float(point['annotations'][0]['x0'])*(256*4)),round(float(point['annotations'][0]['y0'])*(256*4)),round(float(point['annotations'][0]['x1'])*(256*4)),round(float(point['annotations'][0]['y1'])*(256*4)))
        if index%100 == 0:
        	print(index)
        try:

            with _open_file_read_binary(uriInp) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('L')

                img = img.resize((4*256,4*256), Image.BICUBIC)
                img = img.crop(crop)
                img = img.resize(size, Image.BICUBIC)
                imgWEdges = img.copy().filter(ImageFilter.FIND_EDGES)
                with file_io.FileIO(uricrop, mode = 'w') as fThumb:
                    img.save(fThumb, "JPEG")
                with file_io.FileIO(uricropEdge, mode = 'w') as fEdge:
                    imgWEdges.save(fEdge, "JPEG")

        except:
            badIndexes.append(index)
    for index in sorted(badIndexes, reverse=True):
        points.pop(index)

    with file_io.FileIO(pointsOutput, mode='w') as pointSaveJson:
        pointSaveJson.write(json.dumps(points))






