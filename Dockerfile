FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source, the src/ modules it imports (segmentation, forecasting), and data
COPY backend/ ./
COPY src/ ./src/
COPY data/ ./data/

# model.py reads this; the CSV is copied to /app/data above
ENV DATA_PATH=/app/data/zomato_data.csv

# Use gunicorn for production
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
