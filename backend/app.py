import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.predictor import predict_image
from backend.shap_utils import generate_shap, processor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static/", StaticFiles(directory="static"), name="static"
)


@app.get("/")
async def root():
    return {"message": "Hey there fam"}


@app.post("/predict")
async def predict(file: UploadFile = File()):
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        ver, pred = predict_image(temp_path)

        shap, shapim = processor(temp_path)
        shap_img = generate_shap(shap, shapim)

        return {
            "Verdict": ver,
            "Prediction": pred,
            "shap_image": shap_img,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Prediction failed: {str(e)}"},
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
