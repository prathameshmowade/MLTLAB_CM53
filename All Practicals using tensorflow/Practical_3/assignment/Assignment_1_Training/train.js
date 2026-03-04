const sentences = [
  "I love this product",
  "This is amazing",
  "I am very happy",
  "Excellent quality",
  "Very good experience",
  "I like this a lot",
  "Fantastic service",
  "This is wonderful",
  "Highly satisfied",
  "Really good product",

  "I hate this product",
  "This is terrible",
  "Very bad experience",
  "I am unhappy",
  "Worst quality",
  "I dislike this",
  "Awful service",
  "This is horrible",
  "Very disappointing",
  "Extremely bad product"
];

const labels = [
  1,1,1,1,1,1,1,1,1,1,
  0,0,0,0,0,0,0,0,0,0
];

const vocab = [
  "love","amazing","happy","excellent","good","like","fantastic","wonderful",
  "satisfied","really","highly",

  "hate","terrible","bad","unhappy","worst","dislike","awful","horrible",
  "disappointing","extremely"
];

const vectorize = text =>
  vocab.map(word => text.toLowerCase().includes(word) ? 1 : 0);

const xs = tf.tensor2d(sentences.map(vectorize));
const ys = tf.tensor2d(labels, [labels.length, 1]);

const model = tf.sequential();
model.add(tf.layers.dense({
  units: 16,
  activation: "relu",
  inputShape: [vocab.length]
}));
model.add(tf.layers.dense({
  units: 8,
  activation: "relu"
}));
model.add(tf.layers.dense({
  units: 1,
  activation: "sigmoid"
}));

model.compile({
  optimizer: tf.train.adam(0.01),
  loss: "binaryCrossentropy",
  metrics: ["accuracy"]
});

async function trainModel() {
  await model.fit(xs, ys, {
    epochs: 80,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} | Loss: ${logs.loss.toFixed(4)} | Accuracy: ${logs.acc.toFixed(4)}`
        );
      }
    }
  });
  console.log("Training Complete");
}

trainModel();
