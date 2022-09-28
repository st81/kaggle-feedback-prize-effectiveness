FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 AS nvidia

RUN apt-get update -y
RUN apt-get install -y python3.7 python3-pip build-essential
RUN apt-get install -y unzip libsm6 libxrender1 libxext-dev vim git

ADD ./requirements.txt /tmp
RUN pip install --upgrade pip && pip install --upgrade setuptools && pip3 install wheel
RUN pip3 install -r /tmp/requirements.txt 

WORKDIR /kaggle/working