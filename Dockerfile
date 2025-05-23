FROM python:3.9-slim

# Install system dependencies for opencv-python and others
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=process-image.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Activate venv and run Flask
CMD ["/opt/venv/bin/python", "process-image.py"]
