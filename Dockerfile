# syntax=docker/dockerfile:1
# Trigger

FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "API/spark_api.py"]
CMD ["streamlit", "run", "Dashboard/streamlit_dashboard.py"]
