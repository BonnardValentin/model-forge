FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Install required system libraries for production
RUN apt-get install -y libgomp1

# Upgrade pip to avoid potential issues
RUN pip install --upgrade pip

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set default command to run the FastAPI server with more production-oriented settings
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--timeout-keep-alive", "120"]


