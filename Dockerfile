# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting LLM Study & Agent Platform..."\n\
echo "Available services:"\n\
echo "  - Streamlit Web UI: http://localhost:8501"\n\
echo "  - FastAPI Server: http://localhost:8000"\n\
echo ""\n\
if [ "$1" = "streamlit" ]; then\n\
    echo "🌐 Starting Streamlit..."\n\
    streamlit run deployment/app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "api" ]; then\n\
    echo "🔌 Starting FastAPI..."\n\
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000\n\
else\n\
    echo "📚 Available commands:"\n\
    echo "  docker run -p 8501:8501 llm-platform streamlit"\n\
    echo "  docker run -p 8000:8000 llm-platform api"\n\
    echo "  docker run -it llm-platform bash"\n\
fi\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["streamlit"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Labels for metadata
LABEL maintainer="LLM Study Group"
LABEL description="LLM Study & Agent Development Platform"
LABEL version="0.1.0"