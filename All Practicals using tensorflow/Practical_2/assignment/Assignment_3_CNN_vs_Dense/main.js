const output = document.getElementById("output");

function generateData(samples) {
  const xs = tf.randomUniform([samples, 28, 28, 1]);
  const labels = tf.randomUniform([samples], 0, 10, "int32");
  const ys = tf.oneHot(labels, 10);
  return { xs, ys };
}

/* ================= DENSE MODEL ================= */
async function trainDense() {
  output.innerText = "Training Dense Model...\n";

  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  const data = generateData(1000);

  await model.fit(data.xs, data.ys, {
    epochs: 5,
    callbacks: {
      onEpochEnd: (e, l) =>
        output.innerText += `Dense Epoch ${e + 1} Accuracy: ${l.acc.toFixed(4)}\n`
    }
  });

  output.innerText += "\nDense Model Training Complete\n";
}

/* ================= CNN MODEL ================= */
async function trainCNN() {
  output.innerText = "Training CNN Model...\n";

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3,
    activation: "relu"
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  const data = generateData(1000);

  await model.fit(data.xs, data.ys, {
    epochs: 5,
    callbacks: {
      onEpochEnd: (e, l) =>
        output.innerText += `CNN Epoch ${e + 1} Accuracy: ${l.acc.toFixed(4)}\n`
    }
  });

  output.innerText += "\nCNN Model Training Complete\n";
}
