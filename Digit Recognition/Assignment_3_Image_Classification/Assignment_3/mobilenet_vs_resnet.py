import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions
from tensorflow.keras.preprocessing import image

image_folder = "image"
images = os.listdir(image_folder)

mobilenet = MobileNet(weights="imagenet")
resnet = ResNet50(weights="imagenet")

print("\nComparing MobileNet vs ResNet50\n")

for img_name in images:
    img_path = os.path.join(image_folder, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x_mobile = mobilenet_preprocess(x.copy())
    x_resnet = resnet_preprocess(x.copy())

    pred_mobile = mobilenet.predict(x_mobile)
    pred_resnet = resnet.predict(x_resnet)

    mobile_label = decode_predictions(pred_mobile, top=1)[0][0][1]
    resnet_label = decode_predictions(pred_resnet, top=1)[0][0][1]

    print(f"Image: {img_name}")
    print(f"MobileNet Prediction: {mobile_label}")
    print(f"ResNet50 Prediction: {resnet_label}")
    print("-" * 40)
