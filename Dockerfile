# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to keep Python stdout unbuffered and prevent writing pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /app

# Install minimal system dependencies required for FAISS and compiling libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY . .

# Expose port 8000 for FastAPI API access
EXPOSE 8000

# Set default startup command to launch FastAPI server
ENTRYPOINT ["python", "app.py"]
CMD ["--model-dir", "cricket-commentary-model", "--data-dir", "data"]
