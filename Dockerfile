# syntax=docker/dockerfile:1

#FROM python:3.11.9-slim-bullseye
FROM debian:12

WORKDIR /app

RUN apt update
RUN apt install -y python3
RUN python3 --version
#RUN apt install -y python3-pip
#RUN pip3 --version
RUN apt install -y python3.11-venv
RUN python3 -m venv venv

COPY requirements.txt requirements.txt
RUN venv/bin/pip3 install -r requirements.txt

COPY model0802.h5 prediction.jpg server.py .

#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD [ "venv/bin/python3", "server.py" ]

