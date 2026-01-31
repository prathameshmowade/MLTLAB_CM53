import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from models.dense_model import build_dense_model
from models.rnn_model import build_rnn_model

# Load dataset
data = pd.read_csv("dataset/sentiment_data.csv")

texts = data["text"].values
labels = data["label"].map({"positive": 1, "negative": 0}).values

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_len = 10
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

vocab_size = 1000

# Dense Model
dense_model = build_dense_model(vocab_size, max_len)
dense_model.fit(X_train, y_train, epochs=10, verbose=1)
dense_acc = dense_model.evaluate(X_test, y_test, verbose=0)[1]

# RNN Model
rnn_model = build_rnn_model(vocab_size, max_len)
rnn_model.fit(X_train, y_train, epochs=10, verbose=1)
rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)[1]

print("Dense Model Accuracy:", dense_acc)
print("RNN Model Accuracy:", rnn_acc)
