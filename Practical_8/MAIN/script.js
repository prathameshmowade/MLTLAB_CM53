let webcam;
let net;
let classifier;
let dataset = {};
let labels = [];
let interval;

// Setup Webcam
async function setupWebcam() {
  const webcamElement = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  });
  webcamElement.srcObject = stream;

  return new Promise(resolve => {
    webcamElement.onloadedmetadata = () => {
      resolve(webcamElement);
    };
  });
}

// Load MobileNet
async function loadModel() {
  net = await mobilenet.load();
  console.log("MobileNet loaded");
}

// Init
async function init() {
  webcam = await setupWebcam();
  await loadModel();
}
init();

// Add Class
function addClass() {
  const className = document.getElementById('className').value.trim();
  if (!className) return;

  labels.push(className);
  dataset[className] = [];

  const div = document.createElement('div');
  div.innerText = `✔ ${className}`;
  document.getElementById('classList').appendChild(div);

  document.getElementById('className').value = "";
}

// Capture
function capture() {
  if (labels.length === 0) {
    alert("Add class first!");
    return;
  }

  const currentClass = labels[labels.length - 1];
  const activation = net.infer(webcam, true);

  dataset[currentClass].push(activation);

  console.log("Captured:", currentClass);
}

// Train
async function train() {
  if (labels.length === 0) {
    alert("Add classes first!");
    return;
  }

  let xs = [];
  let ys = [];

  labels.forEach((label, index) => {
    dataset[label].forEach(data => {
      xs.push(data);
      ys.push(index);
    });
  });

  const xsTensor = tf.concat(xs);
  const ysTensor = tf.oneHot(tf.tensor1d(ys, 'int32'), labels.length);

  classifier = tf.sequential();

  classifier.add(tf.layers.dense({
    inputShape: [xsTensor.shape[1]],
    units: 100,
    activation: 'relu'
  }));

  classifier.add(tf.layers.dense({
    units: labels.length,
    activation: 'softmax'
  }));

  classifier.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  document.getElementById("results").innerText = "Training... ⏳";

  await classifier.fit(xsTensor, ysTensor, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById("results").innerText =
          `Epoch ${epoch + 1} | Loss: ${logs.loss.toFixed(4)}`;
      }
    }
  });

  document.getElementById("results").innerText = "✅ Training Complete!";
}

// Predict
function predict() {
  if (!classifier) {
    alert("Train model first!");
    return;
  }

  if (interval) clearInterval(interval);

  interval = setInterval(async () => {
    const activation = net.infer(webcam, true);
    const prediction = classifier.predict(activation);

    const result = await prediction.data();
    const maxIndex = result.indexOf(Math.max(...result));

    document.getElementById("results").innerText =
      `Prediction: ${labels[maxIndex]} | Confidence: ${(result[maxIndex] * 100).toFixed(2)}%`;

    const bar = document.getElementById("confidenceBar");
    if (bar) {
      bar.value = result[maxIndex] * 100;
    }

  }, 1000);
}