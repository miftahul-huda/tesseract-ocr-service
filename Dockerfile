FROM ubuntu:18.04


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv -y
RUN apt-get install python3-tk -y
RUN apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn -y
# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN  pip install scikit-build
RUN pip install -r requirements.txt
RUN pip install --upgrade google-cloud-vision
RUN pip install requests



ENV APPLICATION_PORT 8080
ENV GCP_PROJECT lv-tennant-spindo
ENV GCP_UPLOAD_BUCKET lv-tennant-spindo-upload-bucket
ENV GCP_UPLOAD_FOLDER images
#ENV UPLOADER_API https://gcsfileuploader-v2-dot-lv-tennant-spindo.et.r.appspot.com
ENV UPLOADER_API https://gcsmanagerapi-dot-lv-saas.et.r.appspot.com
EXPOSE 8080

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app

#CMD ["python3", "main.py"]

