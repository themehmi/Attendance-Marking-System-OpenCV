# Upgrade to Python 3.10 to satisfy package requirements
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR themehmi/Attendance-Marking-System

# IMPORTANT: Install system dependencies required by OpenCV, dlib, and pip
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first (for caching purposes)
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

EXPOSE 7860

# Command to run your attendance system
CMD ["python", "app.py"]