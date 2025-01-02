import io

import keras
import numpy as np
from keras import ops
from PIL import Image
from keras._tf_keras.keras.models import Model


# from keras.api.utils import
# import pandas as pd


class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, img, size=(224, 224)):
        input_image = Image.open(io.BytesIO(img)).convert("RGB").resize(size)
        # width, height = input_image.size
        # resize_factor = min(size[0] / width, size[1] / height)
        # resized_image = input_image.resize(
        #     (
        #         int(input_image.width * resize_factor),
        #         int(input_image.height * resize_factor),
        #     )
        # )

        img_data = self.preprocess(input_image)
        loaded_model = keras.models.load_model(self.model_path)

        breeds_names = ["Chow", "Toy-Poddle", "Dingo"]

        preds = loaded_model.predict(img_data)
        print("preds")
        print(preds)
        id_prediction = np.argmax(preds)
        proba = preds[0][id_prediction]
        breed = breeds_names[id_prediction]
        print(proba, breed)

        return proba, breed

    def preprocess(self, img):
        img_array = keras.utils.img_to_array(img)
        return keras.ops.expand_dims(img_array, 0)
