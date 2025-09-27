# The tflite-runtime 2.14 library is only available in Python 3.10 and 3.11
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY counting-model0926.tflite classification-model0927.tflite server.py .

# Run the server
CMD [ "python", "server.py" ]
