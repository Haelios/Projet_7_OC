# syntax=docker/dockerfile:1
# Trigger

FROM python:3.9-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install glibc


COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY API/lgb_opti.pkl API/spark_api.py ./
COPY Data/data_train_sample.csv Data/data_test_sample.csv ./

EXPOSE 80

CMD ["python", "spark_api.py"]
