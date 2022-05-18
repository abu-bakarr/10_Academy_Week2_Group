FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY inference.py ./inference.py

# RUN python3 dashboard.py