FROM tensorflow/tensorflow:1.15.0-py3

RUN pip install scikit-image=0.14.2 matplotlib tifffile czifile==2019.7.2 nd2reader==3.2.3

COPY . /app
