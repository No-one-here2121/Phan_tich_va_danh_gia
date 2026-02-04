FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to match volume mount
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy patch script and apply it
COPY patch_vnai.py .
RUN python3 patch_vnai.py

# Copy application code
COPY . .

# Create wait-for-db script
RUN echo '#!/bin/sh\n\
set -e\n\
echo "Waiting for MySQL to be ready..."\n\
until nc -z -v -w30 db 3306; do\n\
  echo "Waiting for database connection..."\n\
  sleep 2\n\
done\n\
echo "MySQL is up and running!"\n\
exec "$@"' > /wait-for-db.sh && \
    chmod +x /wait-for-db.sh

# Expose port
EXPOSE 5000

# Start with wait script
CMD ["/wait-for-db.sh", "python3", "-m", "flask", "run", "--host=0.0.0.0"]