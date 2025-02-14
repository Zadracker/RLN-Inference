# Use official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for torch-points-kernels
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential cmake \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch first to avoid conflicts
RUN pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the test script
CMD ["python", "test.py"]
