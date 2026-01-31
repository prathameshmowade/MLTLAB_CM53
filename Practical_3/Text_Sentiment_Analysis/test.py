import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.rnn_model import build_rnn_model

# Load dataset
data = pd.read_csv("dataset/sentiment_data.csv")

texts = data["text"].values
labels = data["label"].map({"positive": 1, "negative": 0}).values

# Tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Prepare model
max_len = 10
vocab_size = 1000
model = build_rnn_model(vocab_size, max_len)
model.fit(
    pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding='post'),
    labels,
    epochs=50,
    verbose=0
)

# Custom sentences
test_sentences = [
    "I really love this app",
    "This product is very bad",
    "I am not happy with the service",
    "The experience was amazing"
]

# Predict
sequences = tokenizer.texts_to_sequences(test_sentences)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')
predictions = model.predict(padded)

# Output
for sentence, score in zip(test_sentences, predictions):
    sentiment = "Positive" if score > 0.5 else "Negative"
    print(f"Sentence: {sentence}")
    print(f"Confidence Score: {score[0]:.2f}")
    print(f"Predicted Sentiment: {sentiment}\n")
