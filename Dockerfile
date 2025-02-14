# Use official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install main dependencies first
RUN pip install --no-cache-dir -r requirements.txt

# Now install torch-points-kernels separately
RUN pip install torch-points-kernels>=0.5.2

# Command to run the test script
CMD ["python", "test.py"]
