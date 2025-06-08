from fastapi import FastAPI, UploadFile, File, Request,HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
import os

app = FastAPI()

# Define class names
CLASS_NAME = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
    "Spider_mites_Two_spotted_spider_mite", "Target_Spot", "YellowLeaf__Curl_Virus", "mosaic_virus", "healthy"
]

# Mount static directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_models', 'best_model.h5')
MODEL = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return "hello ,aman"

def read_file_as_image(data):
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = read_file_as_image(image_data)
        img_batch = np.expand_dims(image, 0)  # Add batch dimension

        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAME[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
