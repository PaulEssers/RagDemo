# This docker runs the API for the 
# FROM python:3.12-slim
FROM nvidia/cuda:12.0.0-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir \
    transformers \
    torch \
    fastapi \
    uvicorn \
    bitsandbytes \
    accelerate \
    sentencepiece \
    protobuf \
    && apt-get clean

# Copy the FastAPI server script into the container
COPY src/model_server.py /app/model_server.py

# Expose the port the app will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]