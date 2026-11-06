# Python 3.9 slim base image
FROM python:3.9-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/tmp/matplotlib

# Install system dependencies required for audio processing (libsndfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition first for optimal layer caching
COPY requirements.txt /app/

# Install CPU-optimized PyTorch followed by remaining Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Ensure media and temporary directories exist
RUN mkdir -p /app/media /tmp/matplotlib && \
    chmod +x /app/entrypoint.sh

# Expose web service port
EXPOSE 8000

# Health check to verify service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

