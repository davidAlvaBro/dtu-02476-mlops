# Setup environment basics
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Set up ports and environment variables
EXPOSE 9090
ENV PYTHONUNBUFFERED=True

# Set up environment
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy over application
# COPY example.py example.py

# Run model service example
# CMD python example.py
