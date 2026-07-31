"""Count Chocolate II HTTP service: count the chocolate chips in a photo of a cookie.

Endpoints:
    POST /predict   a JPEG -> the chip count + whether it is a cookie at all,
                    saved as a per-prediction folder in the results store
    GET  /health    liveness probe + whether the models loaded

This module is the HTTP layer: it decodes and vets the upload, calls the models,
shapes the response, and saves the artifacts. The models themselves -- loading
them, the preprocessing they expect, and the decode rule for each output -- live
in predictor.py, which imports no web framework so that c6-models' notebooks can
share exactly this preprocessing instead of reimplementing it.

Two TFLite models run per request: a regression CNN that counts chips from a
300x300 image, and a MobileNetV2 classifier that decides cookie / not cookie from a
224x224 one. Below `confidence_threshold` the classification is reported as
"Uncertain". Both model filenames, the threshold, and where results are saved come
from config/<APP_ENV>.yaml -- see config.py.

Errors distinguish what the caller can fix from what they can't, so the UI can say
something better than "something went wrong":

    400  not a JPEG, or not an image at all (also empty or truncated)
    413  the upload is over 20 MB, or decodes to an absurd number of pixels
    422  it isn't a cookie -- the classifier is the gate, so there is no count
    503  the models aren't loaded -- the service is up but can't predict
    500  anything else, i.e. our bug

The 4xx/503 bodies are `{"detail": {"message": ..., "hint": ..., "id": ...}}`:
`message` is safe to show the user, `hint` tells them what to do about it. (FastAPI
answers a request with no `file` field at all with its own 422, whose `detail` is a
list rather than this object -- a client should treat a detail without a `message`
as the generic case.) A 500 says only that something broke on our side and gives
the prediction `id` to quote -- the exception itself goes to the log and to
error.txt in the saved folder, not to the client.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import io
import os

from PIL import Image, UnidentifiedImageError

import config
import predictor
import results_store

# Only JPEG is accepted. The UI says so and its file input filters for it, so this
# is the server agreeing rather than a new restriction -- and the models were
# trained on photos, not screenshots or transparent PNGs.
ACCEPTED_FORMATS = ("JPEG",)

# Refuse anything over 20 MB. Checked before decoding, on the raw bytes: it's the
# cheap test, and it means a huge upload never reaches the decoder. (The body is
# still buffered by the ASGI server before this runs -- capping it earlier would
# take a middleware reading Content-Length, which isn't worth it at this scale.)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Refused(Exception):
    """We won't produce a count, and the caller is the one who can act on it.

    Covers the whole 4xx family: the upload isn't a usable JPEG (400), it's too big
    (413), or it isn't a cookie (422). Carries the status plus a message safe to show
    the user and a hint at what to do about it, so /predict answers with something
    actionable instead of folding every failure into a 500.
    """

    def __init__(self, message, hint, status=400, extra=None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.status = status
        # Anything else the client should get in `detail` -- the 422 uses it to pass
        # the classifier's verdict, so the UI can show its own not-a-cookie dialog
        # instead of a generic error.
        self.extra = extra or {}


@asynccontextmanager
async def lifespan(app):
    """Validate the environment and report it, once, as the app starts serving.

    `config.validate()` fails the boot on a YAML that names a missing model or an
    unknown results backend -- loudly, at startup, rather than on the first request.
    Then the models are actually loaded, so a corrupt .tflite shows up here too
    instead of surfacing as a 503 later.
    """
    config.validate()
    print(f"Environment: APP_ENV={config.APP_ENV}")
    for kind in ("counting", "classification"):
        loaded = predictor.interpreter(kind) is not None
        print(f"Model {kind}: {config.MODELS.get(kind)}"
              f"{'' if loaded else '  [NOT LOADABLE -- /predict will 503]'}")
    print(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"Results store: {config.RESULTS.get('backend', 'local')}")
    yield


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # GET for /health, POST for /predict
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def store():
    """Per-environment results storage (local dir or S3 bucket), built on first use."""
    return results_store.make_store(config.RESULTS)


def decode_upload(contents):
    """The uploaded bytes -> an RGB PIL image, or Refused if they aren't a usable JPEG.

    Everything the caller can get wrong about the *file* is caught here and turned
    into a 400/413, so the only exceptions that escape /predict are genuinely ours.
    `load()` forces the decode to happen now: Image.open is lazy, so a truncated file
    would otherwise blow up later, inside the resize, as an opaque 500.
    """
    if not contents:
        raise Refused("The upload was empty.",
                      "Choose a photo of a cookie and try again.")
    if len(contents) > MAX_UPLOAD_BYTES:
        mb = len(contents) / 1024 / 1024
        raise Refused(f"That file is too large ({mb:.0f} MB).",
                      f"Photos must be under {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
                      status=413)
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Image.DecompressionBombError:
        # Pillow's guard against a small file that decodes to a huge bitmap -- a 20 MB
        # JPEG can still claim enormous dimensions, so the byte cap above doesn't
        # cover this. Same 413: it's too big, just measured in pixels.
        raise Refused("That image is too large to process.",
                      "Resize it and try again.",
                      status=413)
    except UnidentifiedImageError:
        raise Refused("That file isn't an image we can read.",
                      "Upload a JPG photo of a cookie.")
    except OSError as e:
        # Recognized as an image, but the bytes are damaged (the classic being a
        # half-uploaded JPEG: "image file is truncated"). Pillow's wording goes to
        # the log, not into the message -- "6 bytes not processed" means nothing to
        # someone holding a phone.
        print(f"Rejecting a damaged image: {type(e).__name__}: {e}")
        raise Refused("That image file is damaged or incomplete.",
                      "Try uploading it again, or use a different photo.")

    if image.format not in ACCEPTED_FORMATS:
        # `format` is only set by Image.open, and is lost by convert() below, so it
        # has to be read here.
        raise Refused(f"{image.format or 'That file'} images aren't supported.",
                      "Upload a JPG photo of a cookie.")

    # A JPEG can still be greyscale or CMYK; the models were trained on RGB.
    return image.convert("RGB")


def _new_result_id():
    return f"{datetime.now():%Y%m%d}-{hashlib.sha256(os.urandom(16)).hexdigest()[:6]}"


def _start_result(result_id):
    """Open a working directory for this prediction, or None if the store is broken.

    Never raises: saving is for later debugging and ground truth, so a misconfigured
    or unreachable store must not cost the caller their answer -- it costs a warning
    in the log instead.
    """
    try:
        return store().new_working_dir(result_id)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: couldn't open the results store: {e}")
        return None


def _write(out_dir, name, data):
    """Write one artifact into the working directory. Never raises."""
    if out_dir is None:
        return
    try:
        path = out_dir / name
        path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: couldn't save {name}: {e}")


def _finish_result(result_id, out_dir):
    """Finalize the folder (S3: upload it) and return where it landed, or None."""
    if out_dir is None:
        return None
    try:
        store().commit(result_id, out_dir)
        return store().describe(result_id)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: couldn't save this prediction: {e}")
        return None


@app.get("/health")
async def health():
    """Liveness/readiness probe.

    App Runner's health check is currently TCP, which only needs the port open, but
    /predict is a POST -- so if that check is ever switched to HTTP there is
    something for it to call. `ready` reports whether both models loaded.
    """
    return {"ok": True, "models": predictor.describe(), "ready": predictor.ready()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not predictor.ready():
        # 503, not 500: the service is up, it just can't predict -- and no amount of
        # retrying by the caller will change that, so say so plainly.
        raise HTTPException(
            status_code=503,
            detail={
                "message": "The chip counter isn't available right now.",
                "hint": "This is a deployment problem, not yours -- try again later.",
                "detail": f"The models could not be loaded ({predictor.describe()}). "
                          "Train them in c6-models and point config/<APP_ENV>.yaml "
                          "at the .tflite files.",
            },
        )

    result_id = _new_result_id()
    out_dir = _start_result(result_id)
    try:
        contents = await file.read()
        # Save the input before predicting: an image that crashes the pipeline is
        # exactly the one worth keeping.
        _write(out_dir, "input.jpg", contents)

        image = decode_upload(contents)

        # Classify FIRST, and treat it as a gate. A chip count for something that
        # isn't a cookie is a meaningless number, so don't compute one and don't
        # return 200 with a verdict buried in a field the client has to notice.
        verdict, max_confidence, probabilities = predictor.classify(image)
        print(f"Classified {result_id}: {verdict} (confidence {max_confidence:.3f}, "
              f"probabilities {probabilities.tolist()})")

        if verdict != "Cookie":
            _write(out_dir, "prediction.json", json.dumps(
                {"id": result_id, "prediction_text": verdict,
                 "confidence": round(max_confidence, 4)}, indent=2) + "\n")
            raise Refused(
                "That is not a cookie!" if verdict == "Not Cookie"
                else "The count cannot tell whether that is a cookie.",
                "Upload a clear, well-lit photo of a single cookie.",
                status=422,
                extra={"prediction_text": verdict,
                       "confidence": round(max_confidence, 4)},
            )

        chocolate_chips = predictor.count_chips(image)
        payload = {
            "id": result_id,
            # Wrapped in two lists because that is the shape the counting model's
            # output tensor had when this endpoint was written, and c6-www leans on
            # it: `Math.round([[6.94]])` happens to give 7 by way of JS coercing a
            # single-element nested array to a number. Unwrapping it to a plain
            # float would work with today's frontend and read far better -- but it
            # is an API change, not a refactor, so it stays until it's asked for.
            "prediction": [[chocolate_chips]],
            "prediction_text": verdict,
            "confidence": round(max_confidence, 4),
        }
        print(f"Prediction {result_id}: chips={payload['prediction']}")
        _write(out_dir, "prediction.json", json.dumps(payload, indent=2) + "\n")
    except Refused as e:
        # The caller's problem, and one they can act on: say what was wrong and what
        # to do instead. `prediction_text` is only set on the 422 -- it lets the UI
        # show its own not-a-cookie dialog rather than a generic error.
        print(f"Refused {result_id}: {e.status} {e.message}")
        if not e.extra:
            # A rejected *file*, so there is no prediction.json -- record why.
            # (The not-a-cookie 422 already wrote one: its verdict is the result.)
            _write(out_dir, "error.txt", f"{e.status} {e.message}\n")
        raise HTTPException(
            status_code=e.status,
            detail={"message": e.message, "hint": e.hint, "id": result_id, **e.extra},
        )
    except Exception as e:
        # Our problem. The client gets the id to quote and nothing else -- the
        # exception text can name internal paths and model files, and it wouldn't
        # mean anything to whoever uploaded the photo. It goes to the log and to
        # error.txt in the saved folder, which is what the id is for.
        print(f"ERROR: prediction {result_id} failed: {type(e).__name__}: {e}")
        _write(out_dir, "error.txt", f"{type(e).__name__}: {e}\n")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Something went wrong counting the chips.",
                "hint": "Try again; if it keeps happening, quote this id.",
                "id": result_id,
            },
        )
    finally:
        # Runs on both paths, so a failed prediction still leaves its input behind.
        saved_as = _finish_result(result_id, out_dir)

    payload["saved_as"] = saved_as
    return payload


if __name__ == "__main__":
    import uvicorn

    # `reload: true` (local-dev.yaml) auto-restarts the server when a .py file
    # changes. Reload requires the import-string form ("server:app") rather than the
    # app object, and the `watchfiles` package (pip install watchfiles).
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=config.RELOAD)
