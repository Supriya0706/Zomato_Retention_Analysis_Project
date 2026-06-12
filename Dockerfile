FROM python:3.11-slim

WORKDIR /app

# Copy backend code
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source + data
COPY backend/ ./
COPY data/ ./data/

# Use gunicorn for production
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
