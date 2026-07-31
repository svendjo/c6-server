# The tflite-runtime 2.14 library is only available in Python 3.10 and 3.11
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files. Every module server.py imports must be listed: predictor
# (the models and the image handling), config (the env YAML), rate_limit (the
# /predict limiter) and results_store (where predictions are saved). A missing one
# only fails at container start, so keep this in sync.
#
# Nothing needs apt-get for HEIC support: the pillow-heif wheel vendors its own
# libheif, libde265 and libx265, so pip install is the whole story.
COPY server.py predictor.py rate_limit.py config.py results_store.py .
# The two models named in config/aws-prod.yaml's `models:` block. Listed explicitly
# rather than `COPY *.tflite`: c6-models drops every model it trains into this
# directory, and a glob would bake all of them -- including the superseded .h5 and
# .keras sources -- into the image. Yes, the filenames are duplicated with the YAML
# (a retrain means editing both), but that beats an image that grows per experiment.
COPY counting-model0926.tflite classification-model0927.tflite .
COPY config/ ./config/

# Select the production environment: config/aws-prod.yaml drives the confidence
# threshold, the model filenames, and the S3 (chocolate-results) results store.
# The run command stays `python server.py`.
ENV APP_ENV=aws-prod

# Don't buffer stdout. Python block-buffers stdout whenever it isn't a terminal --
# which it never is here -- so the server's print() logging (the startup banner and
# one line per prediction) would sit in an 8KB buffer instead of reaching the App
# Runner log, and be lost outright if the container is killed. Same as `python -u`.
ENV PYTHONUNBUFFERED=1

# Run the server
CMD [ "python", "server.py" ]
