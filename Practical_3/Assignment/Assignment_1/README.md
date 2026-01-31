Title: Training a Sentiment Classifier

Aim:
To train a sentiment classification model using a small dataset of positive and negative sentences.

Dataset Description:
A CSV file containing short text sentences labeled as positive or negative was used. The dataset was preprocessed using tokenization and padding.

Methodology:

Load the dataset

Convert text into numerical sequences using Tokenizer

Pad sequences to equal length

Train a neural network model

Evaluate model accuracy

Model Used:

Dense Neural Network

Loss Function: Binary Crossentropy

Optimizer: Adam

Result:
The model was successfully trained and achieved reasonable accuracy considering the small dataset size.

Conclusion:
The sentiment classifier was able to learn basic sentiment patterns from text data.