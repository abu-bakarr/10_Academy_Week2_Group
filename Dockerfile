FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

COPY models .


RUN pip install -r requirements.txt


COPY train.py ./train.py

# RUN python3 dashboard.py