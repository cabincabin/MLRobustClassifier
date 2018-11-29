
import trainer.model as model


#TO DEPLOY IN GCP ML ENGINE, MUST DELETE ALL LOCAL IMAGE FOLDERS AND 'PointAnnotationsSet"
#TO RUN
#IN GOOGLE CLOUD CONSOLE
#1. >>>REGION=us-central1
#2. >>>JOB_NAME=<GIVEITANAMEANDVERSION>
#3. >>> gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://mlengine_example_bucket --runtime-version 1.8 --module-name trainer.task --package-path trainer/ --region $REGION

if __name__== "__main__":
    model.USEGCP(False) #USE TRUE WHEN DEPLOYING TO GCP ML ENGINE
    model.main()

