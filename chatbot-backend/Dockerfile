FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models data/analyzed

# Expose the port the app runs on
EXPOSE 8000

CMD ["python", "main.py"]
