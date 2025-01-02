import io
import streamlit as st

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
# backend = "http://localhost:8000/api/predict/"
backend = "http://fastapi:8000/api/predict/"


def call_prediction(image, server_url: str):
    encoder = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    request = requests.post(
        server_url,
        data=encoder,
        headers={"Content-Type": encoder.content_type},
        timeout=8000,
    )

    return request


st.title("Prédiction de la race d'un chien.")

st.write(
    """...
         Documentation FastAPI disponible ici `http://localhost:8000/docs`."""
)

input_image = st.file_uploader("Choisissez une image :")

if st.button("Prédire la race"):

    col1, col2 = st.columns(2)

    if input_image:
        # Prediction
        prediction_response = call_prediction(input_image, backend)
        prediction = prediction_response.json()

        original_image = Image.open(input_image).convert("RGB").resize((112, 112))

        col1.header("Image du Chien")
        col1.image(original_image)
        col2.header("Race prédite")
        col2.write(f"Race :  {prediction['breed']}")
        col2.write(f"Proba : {prediction['probability']:.2%}")

    else:
        # handle case with no image
        st.write("Choisissez une image.")
