FROM python:3.10-slim-bullseye

RUN apt update 
RUN apt install -y vim
RUN apt install -y git 
RUN apt install -y graphviz
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

WORKDIR /workspace/from-scratch

COPY ./ ./
