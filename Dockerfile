# CropGuard AI - Docker Container for Always-On Cloud Backend
FROM python:3.11-slim

WORKDIR /app

# Install OpenCV and system graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository
COPY . .

# Default port
ENV PORT=8000
EXPOSE 8000

# Start Uvicorn ASGI server with dynamic port support
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
