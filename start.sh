source myenv/bin/activate
export APPLICATION_PORT=8282 
export GCP_PROJECT=levenshtein-dev
export GCP_UPLOAD_BUCKET=levenshtein-upload-bucket
export GCP_UPLOAD_FOLDER=images
export GOOGLE_APPLICATION_CREDENTIALS=/Users/miftahul.huda/Credentials/levenshtein-dev-ocr-service.json
export UPLOADER_API=https://gcsfileuploader-v2-dot-levenshtein-dev.et.r.appspot.com
python3 main.py 