import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

model = tf.keras.models.load_model("model_cnn.h5")

canvas_size = 280
img = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(img)

def paint(event):
    x, y = event.x, event.y
    draw.ellipse((x-10, y-10, x+10, y+10), fill=255)
    canvas.create_oval(x-10, y-10, x+10, y+10, fill='white')

def predict():
    img_resized = img.resize((28, 28))
    arr = np.array(img_resized).astype("float32")
    arr = arr.reshape(1, 28, 28, 1) / 255.0
    pred = model.predict(arr)
    print("Predicted Digit:", np.argmax(pred))

root = tk.Tk()
root.title("Digit Recognition")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()
canvas.bind("<B1-Motion>", paint)

btn = tk.Button(root, text="Predict", command=predict)
btn.pack()

root.mainloop()
