FROM tensorflow/tensorflow:1.15.0-py3

RUN pip install scikit-image matplotlib tifffile

COPY . /app
