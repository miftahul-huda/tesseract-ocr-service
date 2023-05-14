source myenv/bin/activate
export APPLICATION_PORT=8282 
export GCP_PROJECT=lv-tennant-spindo
export GCP_UPLOAD_BUCKET=lv-tennant-spindo-upload-bucket
export GCP_UPLOAD_FOLDER=images
export GOOGLE_APPLICATION_CREDENTIALS=/Users/mhuda/Works/Credentials/lv-tennant-spindo-owner.json
export UPLOADER_API=https://gcsfileuploader-v2-dot-lv-tennant-spindo.et.r.appspot.com
gcloud config configurations activate lv-tennant-spindo
python3 main.py 