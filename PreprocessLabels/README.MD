For creating the label map and downloading the whole Inclusive Images Set, the tutorial
https://blog.algorithmia.com/deep-dive-into-object-detection-with-open-images-using-tensorflow/
through image downloading was modified to adapt to GCP, and is as follows.
In GCP Cloud Shell:

      # downloads and extracts the openimages bounding box annotations and image path files
      mkdir data
      wget http://storage.googleapis.com/openimages/2017_07/images_2017_07.tar.gz
      tar -xf images_2017_07.tar.gz
      mv 2017_07 data/images
      rm images_2017_07.tar.gz

      wget http://storage.googleapis.com/openimages/2017_07/annotations_human_bbox_2017_07.tar.gz
      tar -xf annotations_human_bbox_2017_07.tar.gz
      mv 2017_07 data/bbox_annotations
      rm annotations_human_bbox_2017_07.tar.gz

      wget http://storage.googleapis.com/openimages/2017_07/classes_2017_07.tar.gz
      tar -xf classes_2017_07.tar.gz
      mv 2017_07 data/classes
      rm classes_2017_07.tar.gz

This downloads all the csv info needed to get the urls to download the images.
We first need to parse the imageurls and reduce duplicates.
The file DeDupl.py does this. 
If you wish to use LESS LABELS then the entire set, create a new file with the wanted class ids separated by a NEW LINE charicter (\n), IE:

      /m/04bcr3
      /m/07j7r
      /m/01g317

download the following file: https://github.com/algorithmiaio/sample-apps/blob/master/deep_dive_demos/open_images_detection/preprocessing/process_metadata.py
and Run the following command:

    python process_metadata.py --annotations_input_path bbox_annotations/2017_07/test/annotations-human-bbox.csv --image_index_inputt_path images/2017_07/test/images.csv --image_index_output_path imagesEdit/2017_07/test/imagesSet.csv --trainable_classes_path classes/2017_07/classes-bbox-trainable.txt --point_output_path imagesEdit/2017_07/test/PointAnnotationsSet.txt
    
Replacing the --trainable_classes_path if needed with the filepath to the LAbELSET created above.
This will create 2 files, the first is a file called imagesSet.csv. This is a json file containing a paired image ID and url. The second is PointAnnotationsSet.txt, which is a JSON file conntainting an array of points: each point contains an ID and an annotation, which is an array of dicts that contains the bounding boxes and labels associated with the ID.

To start uploading the files run the following in the cloud shell:

      curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
      python get-pip.py --user
      pip install tqdm --user
      pip install futures --user
      gcloud auth login
      
and log into the google account with owner access to the current project. 

Create a new screen and run the downloader by:

      screen
      python DownloadParallel.py --images_path imagesEdit/2017_07/test/imagesSet.csv --images_output_directory gs://<BUCKETNAME>/<IMAGESFOLDER>
      
Detach from the screen by
      
      ctl+a d

Check the bucket and confirm the images are downloading. After a long couple hours you will be able to start on any feature engineering wanted before training the models
