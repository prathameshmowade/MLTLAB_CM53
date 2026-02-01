const vocab = [
  "love","amazing","happy","good","excellent",
  "bad","hate","terrible","worst","not"
];

const vectorizeDense = text =>
  vocab.map(w => text.toLowerCase().includes(w) ? 1 : 0);

const vectorizeRNN = text =>
  text.toLowerCase().split(" ").map(w => vocab.indexOf(w) + 1).slice(0,5);

// -------- Dense Model --------
const denseModel = tf.sequential();
denseModel.add(tf.layers.dense({ units: 8, activation: "relu", inputShape: [vocab.length] }));
denseModel.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
denseModel.compile({ optimizer: "adam", loss: "binaryCrossentropy" });

// -------- RNN Model --------
const rnnModel = tf.sequential();
rnnModel.add(tf.layers.embedding({
  inputDim: vocab.length + 1,
  outputDim: 8,
  inputLength: 5
}));
rnnModel.add(tf.layers.simpleRNN({ units: 8 }));
rnnModel.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
rnnModel.compile({ optimizer: "adam", loss: "binaryCrossentropy" });

async function trainModels() {

  const sentences = [
    "i love this",
    "this is good",
    "i am happy",
    "excellent work",

    "i hate this",
    "this is bad",
    "worst experience",
    "terrible service",

    "not good",
    "not happy",
    "not excellent",
    "not bad"
  ];

  // Correct sentiment labels
  const labels = [
    1,1,1,1,
    0,0,0,0,
    0,0,0,1
  ];

  const denseX = tf.tensor2d(sentences.map(vectorizeDense));

  const rnnX = tf.tensor2d(
    sentences.map(s => {
      const v = vectorizeRNN(s);
      while (v.length < 5) v.push(0);
      return v;
    })
  );

  const y = tf.tensor2d(labels, [labels.length, 1]);

  await denseModel.fit(denseX, y, { epochs: 120, verbose: 0 });
  await rnnModel.fit(rnnX, y, { epochs: 120, verbose: 0 });

  console.log("Models trained successfully");
}

trainModels();

async function compareModels() {
  const text = document.getElementById("inputText").value.toLowerCase();

  // Dense prediction
  const denseInput = tf.tensor2d([vectorizeDense(text)]);
  const denseScore = (await denseModel.predict(denseInput).data())[0];

  // RNN prediction
  let rnnVec = vectorizeRNN(text);
  while (rnnVec.length < 5) rnnVec.push(0);
  const rnnInput = tf.tensor2d([rnnVec]);
  const rnnScore = (await rnnModel.predict(rnnInput).data())[0];

  document.getElementById("denseResult").innerHTML =
    `Dense Model: ${denseScore > 0.5 ? "Positive 😊" : "Negative 😞"} (${denseScore.toFixed(2)})`;

  document.getElementById("rnnResult").innerHTML =
    `RNN Model: ${rnnScore > 0.5 ? "Positive 😊" : "Negative 😞"} (${rnnScore.toFixed(2)})`;
}
