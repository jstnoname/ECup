FROM python:3.12.13-slim
LABEL authors="Alex"
WORKDIR /baseline

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["python", "-u", "run.py"]