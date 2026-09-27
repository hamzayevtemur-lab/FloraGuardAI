# Production Dockerfile for FloraGuard AI
# Compatible with Hugging Face Spaces (Port 7860), Render, Railway, and local Docker

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-optimized PyTorch and dependencies
COPY requirements.txt .

# Install CPU PyTorch first to reduce image size and memory usage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and reports
COPY app/ /app/app/
COPY src/ /app/src/
COPY reports/ /app/reports/
COPY models/ /app/models/
COPY scripts/ /app/scripts/

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Launch FastAPI via Uvicorn on dynamic PORT (default 7860 for Hugging Face)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
