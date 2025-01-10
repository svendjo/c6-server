FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model0802.h5 classification-model0105.keras server.py .

# Run the server
CMD [ "python", "server.py" ]
