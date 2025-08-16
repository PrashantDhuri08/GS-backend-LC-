# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir helps keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the embedding model ---
# Copy the download script and run it. This caches the model in the image layer.
COPY download_model.py .
RUN python download_model.py

# Copy the rest of your application code into the container at /app
COPY . .

# Command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "app:app"]