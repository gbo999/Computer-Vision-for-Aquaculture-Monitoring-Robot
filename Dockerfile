FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for Google Drive access
RUN pip install --no-cache-dir gdown

# Create data directories
RUN mkdir -p /data/imagej_measurements \
    /data/molt_exuviae \
    /data/drone_detection

# Copy scripts
COPY scripts/ ./scripts/
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/data

# Default command
CMD ["bash"]
