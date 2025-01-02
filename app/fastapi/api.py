import pathlib
# from typing import Annotated

from models import BreedPrediction
from fastapi import FastAPI, UploadFile, File

from predictor.predictor import Predictor

app = FastAPI(
    title="API classification race de chiens.",
    description="API de classification de race de chiens.",
    version="0.1.2",
)

# predictor = Predictor(pathlib.Path('./predictor/models/inception_v3-2048.keras'))
predictor = Predictor(pathlib.Path('./predictor/models/vgg16-prod.keras'))


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de classification de chiens."}


@app.post("/api/predict/", response_model=BreedPrediction)
async def predict(file: bytes = File(...)):
    proba, breed = predictor.predict(file)
    print("predictions ", proba, breed)

    breedPrediction = BreedPrediction(breed=breed, probability=proba)
    return breedPrediction

    # return {"filename": file.filename,
    #         "file_size": file,
    #         }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.fastapi.api:app", host='0.0.0.0', port=8000, reload=True)
