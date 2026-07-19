FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create upload directory and set write permissions
RUN mkdir -p /code/uploads && chmod 777 /code/uploads

# Copy all source files
COPY . /code

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to launch the web application
CMD ["python", "app.py"]
