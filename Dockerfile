FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN pip install scikit-image matplotlib tifffile czifile==2019.7.2 nd2reader==3.2.3

COPY . /app
RUN curl -f -o /app/models/nucleiDAPILAMIN/model.ckpt.data-00000-of-00001 https://mcmicro.s3.amazonaws.com/models/unmicst2/model.ckpt.data-00000-of-00001
RUN curl -f -o /app/models/nucleiDAPI1-5/model.ckpt.data-00000-of-00001 https://mcmicro.s3.amazonaws.com/models/unmicst1-5/model.ckpt.data-00000-of-00001
