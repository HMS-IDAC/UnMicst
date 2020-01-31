ARG TF_IMAGE=tensorflow/tensorflow:1.15.0-py3
FROM $TF_IMAGE

RUN pip install scikit-image matplotlib tifffile

COPY . /app
