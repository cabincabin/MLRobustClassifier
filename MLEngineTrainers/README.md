## How to run the project files on Google Cloud Machine Learning Engine (Cloud ML Engine)

GCP's Cloud ML Engine was used to execute our CNN training model code on Google Cloud infrastructure. Due to the quite complicated nature that is Big Data and Cloud Computing, there was a tremendous amount of configuration, research, and troubleshooting put into this portion of the project. We hope that the steps below will help those who wish to understand how we executed the code contained in "TrainerTestAndSaveKaggleTune" and "oldClaireTrainerSaveData".

To learn more about ML Engine:
- Getting Started guide by GCP: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
- ML Engine Examples to try out (the team focused on the "mnist" and "flowers" examples): https://github.com/GoogleCloudPlatform/cloudml-samples

## Steps to run oldClaireTrainerSaveData
The steps below assume there exists a GCS (Google Cloud Storage) bucket that contains the input images and "PointAnnotationsSet" text file. 

### Zip contents and upload to Cloud Shell or Clone from git
Within your local file directory, zip the folder "oldClaireTrainerSaveData" in a .zip file. Then log into the [GCP Console web dashboard](https://console.cloud.google.com/home/), and click "Activate Cloud Shell" in the top-right corner. In this Terminal console window, click the "3 vertical dots" icon and select "Upload File". Upload the .zip file and unzip the contents. Then execute the following shell commands.

```shell
ls
cd oldClaireTrainerSaveData/
```

### Install dependencies
Install the python dependencies via: `pip install --user -r requirements.txt`

### Modify the files to suit your needs
In trainer/model.py, if you created a custom number of labels, change line 134

     model.add(Dense(500, activation='softmax'))
     
to have the number of labels specified in the custom labels

in trainer/task.pyc line 142, change the output line to the output path to your chosen test output path. This is where the results of testing will end up.  

     model.main(ImageLabelDict, getIDLabelDict(points), SuccessNum, "gs://mlclassifiertuning/imageIds/edgeResultsLabel/")
     
goes to

     model.main(ImageLabelDict, getIDLabelDict(points), SuccessNum, "gs://<BUCKETNAME>/<PATHTOTESTRESULTS>/")

### Configuration Variables
Run the following shell script within the Cloud Shell:

```shell
BUCKET_NAME=<METADATABUCKET>
REGION=us-central1
JOB_NAME=<CHOOSE_JOB_NAME>
OUTPUT_PATH=gs://$BUCKET_NAME
POINTS_PATH=gs://<BUCKET>/<PATHTO>/PointAnnotationsSet256x256.txt
IMAGES_INPUT=gs://<BUCKET>/<PATHTOFEATUREENGINEEREDIMAGES>/
```

### Run a training job in the cloud
Run the following shell script within the Cloud Shell:

```shell
gcloud ml-engine jobs submit training $JOB_NAME \
--staging-bucket $OUTPUT_PATH \
--runtime-version 1.10 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
-- \
--points-path $POINTS_PATH \
--images-input-path $IMAGES_INPUT
```

NOTE: If you want to make use of the GPU computing power in the Google Cloud, then you would modify `config.yaml` and have an argument called `--config config.yaml` before the `--region $REGION` argument. WARNING, GPU power is EXPENSIVE, make sure you have enough GCP Credits/can pay out of pocket. More details on config.yaml can be found here: https://cloud.google.com/ml-engine/docs/tensorflow/machine-types
