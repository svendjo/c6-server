"""The two models: cookie / not cookie, and how many chips are in it.

Everything between a PIL image and a number lives here -- loading the TFLite
interpreters, the preprocessing the models were trained on, and the decode rule
for each output. Imports no web framework, so a notebook can use it without
pulling in FastAPI:

    import sys, os
    sys.path.insert(0, os.path.abspath("../c6-server"))
    import predictor

    verdict, confidence, _ = predictor.classify(img)
    chips = predictor.count_chips(img)

That is the point of the module. Both preprocessing and the classifier's
threshold rule used to exist twice -- once here and once in c6-models' notebooks
-- and they drifted: the notebooks resize with Keras' default NEAREST while the
server used Pillow's default BICUBIC, so the models were served images they were
never trained on. One definition, imported by both, is what stops that
recurring.

Model filenames and the confidence threshold come from config/<APP_ENV>.yaml.
"""
from functools import lru_cache

try:
    from tflite_runtime.interpreter import Interpreter as tflite
except ImportError:
    # Production installs the small tflite-runtime wheel; a dev machine has full
    # TensorFlow (requirements-dev.txt), whose bundled interpreter is API-compatible
    # for what we do here. tf.lite is lazily loaded, so it must be reached via
    # attribute access -- `from tensorflow.lite import Interpreter` does not trigger
    # the lazy loader and fails on TF 2.16 (as the previous fallback import did).
    import tensorflow as tf
    tflite = tf.lite.Interpreter
from PIL import Image
import numpy as np

import config

CLASS_LABEL_BY_ID = {0: "Not Cookie", 1: "Cookie"}

# Each model's expected input side length, in pixels. The models are square and
# take NHWC float32 in [0, 1]; these have to match how they were trained in
# c6-models (Counting.ipynb at 300, Classification.ipynb at 224).
INPUT_SIZE = {"counting": (300, 300), "classification": (224, 224)}

# How the training images were resized, and therefore how ours must be. Both
# notebooks go through Keras (`flow_from_dataframe`, `load_img`), whose default
# interpolation is "nearest" -- while Pillow's `resize` defaults to BICUBIC. The
# server used the bare default for a long time and so fed both models a smoother
# image than either was trained on; across the 62 dataset photos that moved the
# chip count by 0.70 on average and up to 1.78. Matching training is the cheap
# fix. Switching both sides to bicubic would be defensible -- nearest-sampling a
# phone photo down to 300x300 aliases badly -- but that is a retrain, not a
# constant.
RESAMPLE = Image.Resampling.NEAREST


@lru_cache(maxsize=None)
def interpreter(kind):
    """The allocated TFLite interpreter for "counting" / "classification", or None.

    Loaded on first use and cached, rather than at import: a missing or corrupt
    model file then becomes a 503 with a message instead of a stack trace that
    kills the process, and importing this module (e.g. from a notebook) costs
    nothing. Returns None on failure -- `ready()` is what callers gate on.
    """
    try:
        interp = tflite(model_path=str(config.model_path(kind)))
        interp.allocate_tensors()
        return interp
    except Exception as e:  # noqa: BLE001 -- report, don't crash the process
        print(f"WARNING: couldn't load the {kind} model: {e}")
        return None


def ready():
    """True when both models are loaded -- a prediction needs them both."""
    return all(interpreter(k) is not None for k in ("counting", "classification"))


def describe():
    return ", ".join(f"{k}={config.MODELS.get(k)}"
                     for k in ("counting", "classification"))


def preprocess_image(image, target_size):
    """A PIL image -> the NHWC float32 batch of one the models were trained on."""
    image = image.resize(target_size, RESAMPLE)
    image = np.array(image, dtype=np.float32) / 255.0  # normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # add the batch dimension


def infer(kind, image):
    """Run `image` through the named model and return its raw output tensor."""
    interp = interpreter(kind)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]["index"],
                      preprocess_image(image, INPUT_SIZE[kind]))
    interp.invoke()
    return interp.get_tensor(output_details[0]["index"])


def decide_verdict(probabilities, confidence_threshold=None):
    """A row of class probabilities -> (verdict, confidence).

    `verdict` is "Cookie", "Not Cookie", or "Uncertain" when the winning class is
    below `confidence_threshold` -- the classifier isn't sure enough to commit
    either way, and saying so is more useful than a coin flip.

    Deliberately takes probabilities rather than an image, so that it is the one
    definition of the rule for both the deployed TFLite model and a Keras model
    still in memory in Classification.ipynb. The threshold defaults to the
    configured one but can be overridden, which is what the notebook does when it
    sweeps it.
    """
    if confidence_threshold is None:
        confidence_threshold = config.CONFIDENCE_THRESHOLD
    max_confidence = float(np.max(probabilities))
    class_id = int(np.argmax(probabilities))
    verdict = (CLASS_LABEL_BY_ID[class_id]
               if max_confidence >= confidence_threshold else "Uncertain")
    return verdict, max_confidence


def classify(image):
    """Is this a cookie? -> (verdict, confidence, probabilities).

    Runs the deployed classifier and applies `decide_verdict`. `probabilities` is
    the raw output row, returned for logging rather than for a decision.
    """
    probabilities = infer("classification", image)[0]
    verdict, max_confidence = decide_verdict(probabilities)
    return verdict, max_confidence, probabilities


def count_chips(image):
    """How many chocolate chips are in this cookie? -> a float.

    A regression, so the answer is continuous: 6.94 means "about seven". Rounding
    is the caller's business -- the saved artifacts keep the raw value.
    """
    return float(infer("counting", image)[0][0])
