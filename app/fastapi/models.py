from pydantic import BaseModel


class BreedPrediction(BaseModel):
    breed: str
    probability: float
