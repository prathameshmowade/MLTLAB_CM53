import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

model = MobileNet(weights="imagenet")

img_path = os.path.join("images", os.listdir("images")[0])

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

preds = model.predict(img_array)

top3 = decode_predictions(preds, top=3)[0]

print("Top-3 Predictions:")
for i, pred in enumerate(top3):
    print(f"{i+1}. {pred[1]} : {pred[2]*100:.2f}%")
