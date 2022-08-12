FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv


# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


RUN  pip install scikit-build

# Install production dependencies.
RUN pip install -r requirements.txt

ENV APPLICATION_PORT=8282
EXPOSE 8282

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
#CMD exec gunicorn --bind :8282 --workers 1 --threads 8 --timeout 0 main:app

CMD ["python3", "main.py"]
