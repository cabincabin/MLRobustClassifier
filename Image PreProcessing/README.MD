The process for running each of these models is the same. 
The first, trainer for cropping images, will create 2 image sets, one black and white,and one the outlines of the objects.
Both images will be 256 by 256.
the second will do the same, but first cropped to the label's bounding box before resized to 256 by 256.

TO RUN. 
Enable the ML Engine in GCP and make sure your PointAnnotationsSet.txt from the first set of steps is in a location in the
CLOUD BUCKET.

Clone TrainerForCroppingImagesToBBox to the GCP cloud shell.
To use the standard image cropper, rename modelOld.py to model.py and remove the second model.py.
Otherwise, to use the standard BBox cropper, leave the file names as they are.

In lines 17-22 of model.py, rename the filepaths to the custom paths specified by your personal project and filepaths:
 
      subprocess.call(["gsutil", "cp", "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet.txt", "PointAnnotationsSet.txt"])
      global imageInputPath, imageOutputScaled, imageOutputEdge, pointsOutput
      imageInputPath = "gs://wpiopenimageskaggle/ImageFiles/"
      imageOutputScaled = "gs://wpiopenimageskaggle/Imagefilesbbox256/"
      imageOutputEdge = "gs://wpiopenimageskaggle/ImagefilesbboxEdge/"
      pointsOutput = "gs://wpiopenimageskaggle/Imageids/PointAnnotationsSet256bbox.txt"
      
would go to

      subprocess.call(["gsutil", "cp", "gs://<BUCKET>/imageIds/PointAnnotationsSet.txt", "PointAnnotationsSet.txt"])
      global imageInputPath, imageOutputScaled, imageOutputEdge, pointsOutput
      imageInputPath = "gs://<BUCKET>/<PATHTOImageFiles>/"
      imageOutputScaled = "gs://<BUCKET>/Imagefilesbbox256/"
      imageOutputEdge = "gs://<BUCKET>/ImagefilesbboxEdge/"
      pointsOutput = "gs://<BUCKET>/<PATHTO>/PointAnnotationsSet256bbox.txt"

Save and exit the file. create a new bucket if wanted for metadata durring the run. In the cloudshell run

      BUCKET_NAME=<METADATABUCKET>
      REGION=us-central1
      JOB_NAME=<CHOOSE_A_JOB_NAME>
      gcloud ml-engine jobs submit training $JOB_NAME \
      --staging-bucket $OUTPUT_PATH \
      --runtime-version 1.10 \
      --module-name trainer.task \
      --package-path trainer/ \
      --region $REGION \
      
 Confirm the job sucsessfully runs by looking at the ML engine job requests in GCP. THis WILL take a while. 
 From here, you can start training your actual classifier
    
