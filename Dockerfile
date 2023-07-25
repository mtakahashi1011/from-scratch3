FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update 
RUN apt install -y vim
RUN apt install -y git 
RUN apt install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

WORKDIR /workspace

COPY requirements.txt /workspace
COPY src/ /workspace/src/

RUN python3 -m pip install -r requirements.txt
