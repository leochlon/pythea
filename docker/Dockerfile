FROM python:3.11-slim

LABEL maintainer="Glass Team"
LABEL description="Production-ready Glass API for hallucination detection"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Install additional dependencies for Glass
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0

# Copy application code
COPY hallbayes/ /app/hallbayes/
COPY glass/ /app/glass/
COPY benchmarks/ /app/benchmarks/

# Create directories for logs and metrics
RUN mkdir -p /app/logs /app/metrics

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "glass.api:app", "--host", "0.0.0.0", "--port", "8000"]
