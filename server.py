from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from tflite_runtime.interpreter import Interpreter as tflite
except ImportError:
    from tensorflow.lite.interpreter import Interpreter as tflite
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

# Load the counting model
model_interpreter = tflite(model_path='counting-model0926.tflite')
model_interpreter.allocate_tensors()
model_input_details = model_interpreter.get_input_details()
model_output_details = model_interpreter.get_output_details()

# Load the classification model
classification_interpreter = tflite(model_path='classification-model0927.tflite')
classification_interpreter.allocate_tensors()
classification_input_details = classification_interpreter.get_input_details()
classification_output_details = classification_interpreter.get_output_details()

def preprocess_image(image, target_size=(300, 300)):
    image = image.resize(target_size)  # Resize the image to match your model's input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(image.shape)

    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image300 = preprocess_image(image)

        # Run inference on the counting model
        model_interpreter.set_tensor(model_input_details[0]['index'], processed_image300)
        model_interpreter.invoke()
        prediction = model_interpreter.get_tensor(model_output_details[0]['index'])

        # Run inference on the classification model
        processed_image224 = preprocess_image(image, target_size=(224, 224))
        classification_interpreter.set_tensor(classification_input_details[0]['index'], processed_image224)
        classification_interpreter.invoke()
        predictions = classification_interpreter.get_tensor(classification_output_details[0]['index'])[0]
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
