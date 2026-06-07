FROM python:3.11-slim

WORKDIR /app

# Install gcc needed to build twofish (hopsworks dependency)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api.py .
COPY pipelines/mongo_store.py ./pipelines/mongo_store.py

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]