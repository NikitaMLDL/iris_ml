FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY models/model.pkl ./models/model.pkl
CMD ["python", "src/predict.py"]
