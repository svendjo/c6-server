from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('model0802.h5')

# Load the classification model
classification_model = tf.keras.models.load_model('classification-model0105.keras')


def preprocess_image(image, target_size=(300, 300)):
    image = image.resize(target_size)  # Resize the image to match your model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(image.shape)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)

        # Make classification
        processed_image2 = preprocess_image(image, target_size=(224, 224))
        predictions = classification_model.predict(processed_image2)[0]
        print(f"predictions: {predictions}")
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        print(f"predicted_class: {predicted_class}")  # THIS DOESN"T WSORK

        # Determine prediction based on confidence threshold
        confidence_threshold = 0.8

        if max_confidence >= confidence_threshold:
            if predicted_class == 1:
                prediction_text = "Cookie"
            else:
                prediction_text = "Not Cookie"
        else:
            prediction_text = "Uncertain"

        return {"prediction": prediction.tolist(), "prediction_text": prediction_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
