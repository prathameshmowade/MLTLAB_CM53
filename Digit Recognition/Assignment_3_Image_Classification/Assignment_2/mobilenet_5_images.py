import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNet(weights="imagenet")

image_folder = "images"
images = os.listdir(image_folder)

correct = 0
total = len(images)

for img_name in images:
    img_path = os.path.join(image_folder, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    top_pred = decode_predictions(preds, top=1)[0][0][1]

    expected = img_name.split(".")[0].lower()

    print(f"Image: {img_name}")
    print(f"Predicted: {top_pred}")

    if expected in top_pred.lower():
        correct += 1
        print("Result: Correct\n")
    else:
        print("Result: Incorrect\n")

accuracy = (correct / total) * 100
print(f"Final Accuracy on {total} images: {accuracy:.2f}%")
