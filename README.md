# c6-server
Web service for Count Chocolate II (the site is **https://countchocolate.com**).

Takes an uploaded photo of a cookie, runs two TFLite models from `c6-models`, and
returns how many chocolate chips it sees — plus whether the photo is a cookie at
all, so the count isn't reported for a picture of something else. Every prediction
is saved server-side under its own id (`YYYYMMDD-xxxxxx`) with the input image, so a
count someone disagrees with can be reproduced later.

`POST /predict` (multipart `file`) →
```json
{
  "id": "20260730-a1b2c3",
  "prediction": [[12.4]],
  "prediction_text": "Cookie",
  "confidence": 0.9873,
  "saved_as": "results/20260730-a1b2c3"
}
```
A **200 only ever describes a cookie**: the classifier runs first and acts as a gate,
so a photo of something else is refused with **422** and the counting model never
runs (see Errors). `prediction_text` is therefore always `Cookie`; it is kept in the
response because the client keys its dialogs off it. `saved_as` is `null` if the
results store was unreachable — saving never costs the caller their answer, it just
logs a warning.

Uploads must be **JPG, under 20 MB**. The format is checked from the bytes rather
than the filename or content-type, so renaming a PNG doesn't get it past the gate.

`GET /health` → `{"ok": true, "models": "...", "ready": true}`. `ready` is false when
a model file is missing or corrupt.

## Errors
Failures are separated into the caller's and ours, so the UI can say something better
than "something went wrong":

| Status | When | |
|---|---|---|
| **400** | not a JPG, or not an image at all (also empty or truncated) | caller can fix it |
| **413** | the upload is over 20 MB, or decodes to an absurd number of pixels | caller can fix it |
| **422** | **it isn't a cookie** — the classifier is a gate, so there is no count | caller can fix it |
| **422** | no `file` field at all (FastAPI's own validation) | caller can fix it |
| **503** | the models aren't loaded — the service is up but can't predict | deployment |
| **500** | anything else | our bug |

The 4xx/503 bodies are `{"detail": {"message": ..., "hint": ..., "id": ...}}` —
`message` is safe to show the user and `hint` says what to do about it:

```json
{"detail": {"message": "That file isn't an image we can read.",
            "hint": "Upload a JPG photo of a cookie.",
            "id": "20260730-85d461"}}
```

The **not-a-cookie 422** additionally carries `prediction_text` (`"Not Cookie"` or
`"Uncertain"`) and `confidence`, so the UI can show its own not-a-cookie dialog
rather than a generic error. It is a verdict about the image, not a failure — the
classification still gets saved as `prediction.json`. FastAPI's *own* 422 (a request
with no `file` field) is also a 422 but its `detail` is a list; treat a `detail`
without a `message` as the generic case.

A **500** deliberately says only that something broke on our side and gives the `id`
to quote. The exception text can name internal paths and model files and would mean
nothing to whoever uploaded the photo, so it goes to the log and to `error.txt` in
the saved folder instead — which is what the id is for.

## Models
Two models run per request, both on `tflite-runtime` (no full TensorFlow in prod):

- **Counting** (`counting-model0926.tflite`) — a 4-conv-layer CNN *regression* on a
  300×300 image → the chip count as a float.
- **Classification** (`classification-model0927.tflite`) — MobileNetV2 transfer
  learning on a 224×224 image → cookie / not-cookie, trained with CIFAR-10 as the
  negatives.

Both are loaded **lazily on first use** and cached, so a bad model file becomes a 503
with a message rather than an import-time crash. Their filenames live in
`config/<APP_ENV>.yaml`, not in the source — see below.

Everything between a PIL image and a number lives in **`predictor.py`** — loading the
interpreters, the preprocessing the models expect, and the decode rule for each
output (`classify`, `count_chips`, `decide_verdict`). `server.py` is the HTTP layer
around it: it vets the upload, calls those, and shapes the response.

The split exists so that **c6-models' notebooks can import the same preprocessing
instead of reimplementing it** (`sys.path.insert(...); import predictor`, the same
pattern abi-models uses against abi-server). They previously had their own copy and
it drifted: the notebooks resize through Keras, whose default interpolation is
`nearest`, while the server used Pillow's `resize` default of **bicubic** — so both
deployed models were being served images preprocessed unlike anything they were
trained on. Over the 62 dataset photos that moved the predicted chip count by 0.70 on
average and up to 1.78. `predictor.RESAMPLE` is now the single definition, and both
notebooks pass it explicitly to Keras rather than relying on a default.

## Configuration
One YAML per environment in `config/`, selected by the `APP_ENV` environment
variable (default `local-dev`; the Dockerfile sets `aws-prod`):

```yaml
reload: true                # uvicorn auto-restart on .py changes (dev only)
confidence_threshold: 0.8   # below this the classifier answers "Uncertain"
models:
  counting: counting-model0926.tflite
  classification: classification-model0927.tflite
results:
  backend: local            # or s3 (+ bucket, region)
  dir: results
```

So the run command stays `python server.py` in every environment, and a retrained
model is a config change rather than an edit in three files. `config.validate()` runs
in the app's startup lifespan: a YAML naming a missing model, an out-of-range
threshold, or an unknown results backend **fails the boot loudly**, listing every
problem at once, instead of surfacing on the first request.

## Results storage
Each prediction gets `results/<id>/` (local) or `s3://chocolate-results/<id>/`
(prod) holding `input.jpg` and `prediction.json` — or `error.txt` if the prediction
raised, since a crashing input is the one worth keeping. Prod needs S3 because App
Runner's disk is wiped on restart and not shared across instances; that bucket and
its IAM instance role are **not provisioned yet** (see DEPLOY.md part A6).

## Local
Create a virtual environment.

`pyenv virtualenv 3.12.7 venv-c6-server`

Activate the environment.

`pyenv local venv-c6-server`

Install requirements.

`pip install setuptools`
`pip install -r requirements-dev.txt`

Train the models in `c6-models` (run `Counting.ipynb` and `Classification.ipynb`) —
they write their `.tflite` straight into this directory. Then run the server:

`python server.py`

Test it.

`curl -X POST http://localhost:8080/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../c6-models/prediction.jpg"`

`curl http://localhost:8080/health`

## Deployment to AWS
`./deploy.sh` builds the image, pushes it to ECR, and prints the command to trigger
the App Runner deployment. See **[DEPLOY.md](DEPLOY.md)** for the full end-to-end
guide — backend → ECR + App Runner, frontend → S3 + CloudFront, and the one-time
setup of each. Deploy the backend first, then `c6-www`.
