from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import io
import requests
from google.cloud import translate_v2 as translate
import os

app = FastAPI()

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google Translate Client
translate_client = translate.Client()

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(LABELS_URL).text.splitlines()

# Preprocessing
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/recognize")
async def recognize_object(file: UploadFile = File(...), target_lang: str = Form(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_tensor = transform_pipeline(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = outputs.max(1)
    object_name = imagenet_classes[predicted_idx]

    # Translate
    translation = translate_client.translate(object_name, target_language=target_lang)

    return JSONResponse({
        "object_name": object_name,
        "translated_name": translation["translatedText"],
        "language": target_lang
    })
