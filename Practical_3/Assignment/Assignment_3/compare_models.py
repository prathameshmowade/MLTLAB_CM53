import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv("../Assignment_1/dataset/sentiment_data.csv")

sentences = data["sentence"]
labels = data["sentiment"].map({"positive": 1, "negative": 0})

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=10)

dense_model = Sequential([
    Embedding(1000, 16, input_length=10),
    tf.keras.layers.Flatten(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

rnn_model = Sequential([
    Embedding(1000, 16, input_length=10),
    SimpleRNN(16),
    Dense(1, activation="sigmoid")
])

for model, name in [(dense_model, "Dense"), (rnn_model, "RNN")]:
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(f"\nTraining {name} Model")
    model.fit(padded, labels, epochs=5, verbose=1)
