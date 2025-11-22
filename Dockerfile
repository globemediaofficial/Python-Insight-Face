# Use lightweight Python image
FROM python:3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir fastapi uvicorn insightface numpy pillow

# Copy server code
WORKDIR /app
COPY server.py .

# Expose port
EXPOSE 3291

# Run FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3291"]
