const vocab = [
  "love","amazing","happy","excellent","good","like","fantastic","wonderful",
  "satisfied","really","highly",
  "hate","terrible","bad","unhappy","worst","dislike","awful","horrible",
  "disappointing","extremely"
];

const vectorize = text =>
  vocab.map(word => text.toLowerCase().includes(word) ? 1 : 0);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [vocab.length] }));
model.add(tf.layers.dense({ units: 8, activation: "relu" }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

model.compile({
  optimizer: tf.train.adam(0.01),
  loss: "binaryCrossentropy",
  metrics: ["accuracy"]
});

// training data
const trainX = tf.tensor2d([
  vectorize("i love this product"),
  vectorize("this is amazing"),
  vectorize("very happy with service"),
  vectorize("excellent and fantastic"),
  vectorize("i hate this"),
  vectorize("this is terrible"),
  vectorize("worst experience"),
  vectorize("very bad product")
]);

const trainY = tf.tensor2d([[1],[1],[1],[1],[0],[0],[0],[0]]);

(async () => {
  await model.fit(trainX, trainY, {
    epochs: 40,
    verbose: 0
  });
  console.log("Model trained");
})();

async function predictSentiment() {
  const text = document.getElementById("inputText").value;
  if (!text) return;

  const inputTensor = tf.tensor2d([vectorize(text)]);
  const prediction = model.predict(inputTensor);
  const score = prediction.dataSync()[0];

  let sentiment = "";
  if (score > 0.6) sentiment = "Positive 😊";
  else if (score < 0.4) sentiment = "Negative 😞";
  else sentiment = "Neutral 😐";

  document.getElementById("result").innerHTML =
    `Sentiment: ${sentiment}<br>Confidence Score: ${score.toFixed(2)}`;
}
