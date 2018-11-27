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

def GCPPath(UseGCP):
    if UseGCP == True:
        subprocess.call(["gsutil", "cp", "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet.txt", "PointAnnotationsSet.txt"])
        global imageInputPath, imageOutputScaled, imageOutputEdge
        imageInputPath = "gs://wpiopenimageskaggle/ImageFiles/"
        imageOutputScaled = "gs://wpiopenimageskaggle/Imagefiles256x256/"
        imageOutputEdge = "gs://wpiopenimageskaggle/ImagefilesEdge256x256/"


def read_file_JSON(filename):

    with open(filename, 'r') as file:
        # self.response.write(cloudstorage_file.read())
        annotString = file.read()
        file.close()

    annotations = json.loads(annotString)
    return annotations

def _open_file_read_binary(uri):
    try:
        return file_io.FileIO(uri, mode='rb')
    except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')


def getImageGCPPaths():
    global imageInputPath, imageOutputScaled, imageOutputEdge
    points = read_file_JSON("PointAnnotationsSet.txt")
    path = []
    label = []
    for point in points:
        path.append([[imageInputPath + point['id'] + ".jpg"], [imageOutputScaled + point['id'] + ".jpg"], [imageOutputEdge + point['id'] + ".jpg"]])
        label.append(point['label'])

    size = 256,256
    for uri in path:
        try:
            with _open_file_read_binary(uri[0]) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('L')
                img.thumbnail(size, Image.ANTIALIAS)
                imgWEdges = img.copy().filter(ImageFilter.FIND_EDGES)
                with file_io.FileIO(uri[1], mode = 'wb') as fThumb:
                    img.save(fThumb, "JPEG")
                with file_io.FileIO(uri[2], mode = 'wb') as fEdge:
                    img.save(fEdge, "JPEG")

        except:
            path.remove(uri)






