import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

model = tf.keras.models.load_model("../Assignment_1/sentiment_dense_model.h5")

data = pd.read_csv("../Assignment_1/dataset/sentiment_data.csv")
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(data["sentence"])

test_sentences = [
    "I really love this app",
    "This product is very bad",
    "I am not happy with the service",
    "The experience was amazing"
]

sequences = tokenizer.texts_to_sequences(test_sentences)
padded = pad_sequences(sequences, maxlen=10)

predictions = model.predict(padded)

for i, sentence in enumerate(test_sentences):
    score = predictions[i][0]
    sentiment = "Positive" if score > 0.5 else "Negative"
    print(f"Sentence: {sentence}")
    print(f"Confidence Score: {score:.2f}")
    print(f"Predicted Sentiment: {sentiment}\n")
