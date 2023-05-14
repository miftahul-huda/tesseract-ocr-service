source myenv/bin/activate
export APPLICATION_PORT=8282 
export GCP_PROJECT=levenshtein-dev
export GCP_UPLOAD_BUCKET=lv-tennant-spindo-upload-bucket
export GCP_UPLOAD_FOLDER=images
export GOOGLE_APPLICATION_CREDENTIALS=/Users/mhuda/Works/Credentials/levenshtein-dev-ocr-service.json
export UPLOADER_API=https://gcsfileuploader-v2-dot-levenshtein-dev.et.r.appspot.com
gcloud config configurations activate levenshtein-dev
python3 main.py 