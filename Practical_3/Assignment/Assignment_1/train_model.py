import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

data = pd.read_csv("dataset/sentiment_data.csv")

sentences = data["sentence"].values
labels = data["sentiment"].map({"positive": 1, "negative": 0}).values

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=10)

model = Sequential([
    Embedding(1000, 16, input_length=10),
    Flatten(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(padded, labels, epochs=10, verbose=1)

model.save("sentiment_dense_model.h5")
print("Model training completed and saved.")
