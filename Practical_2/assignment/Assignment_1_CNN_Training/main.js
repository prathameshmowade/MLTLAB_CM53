const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const MNIST_IMAGES =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

let images;
let labels;

async function loadMNIST() {
  const img = new Image();
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const imgPromise = new Promise(resolve => {
    img.crossOrigin = "";
    img.onload = () => resolve();
    img.src = MNIST_IMAGES;
  });

  const labelsPromise = fetch(MNIST_LABELS).then(r => r.arrayBuffer());

  const [, labelBuffer] = await Promise.all([imgPromise, labelsPromise]);
  labels = new Uint8Array(labelBuffer);

  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  images = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);

  let p = 0;
  for (let i = 0; i < imageData.data.length; i += 4) {
    images[p++] = imageData.data[i] / 255;
  }

  console.log("MNIST loaded for Assignment-1");
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.reshape({
    inputShape: [784],
    targetShape: [28, 28, 1]
  }));

  model.add(tf.layers.conv2d({ filters: 8, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

function getData() {
  const TRAIN_SIZE = 5000;
  const TEST_SIZE = 1000;

  const xTrain = tf.tensor2d(images.slice(0, TRAIN_SIZE * IMAGE_SIZE), [TRAIN_SIZE, IMAGE_SIZE]);
  const yTrain = tf.tensor2d(labels.slice(0, TRAIN_SIZE * NUM_CLASSES), [TRAIN_SIZE, NUM_CLASSES]);

  const xTest = tf.tensor2d(
    images.slice(TRAIN_SIZE * IMAGE_SIZE, (TRAIN_SIZE + TEST_SIZE) * IMAGE_SIZE),
    [TEST_SIZE, IMAGE_SIZE]
  );
  const yTest = tf.tensor2d(
    labels.slice(TRAIN_SIZE * NUM_CLASSES, (TRAIN_SIZE + TEST_SIZE) * NUM_CLASSES),
    [TEST_SIZE, NUM_CLASSES]
  );

  return { xTrain, yTrain, xTest, yTest };
}

async function run() {
  await loadMNIST();

  const model = createModel();
  model.summary();

  const { xTrain, yTrain, xTest, yTest } = getData();

  console.log("Training Assignment-1 CNN...");

  await model.fit(xTrain, yTrain, {
    epochs: 5,
    batchSize: 128,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} Accuracy: ${(logs.acc * 100).toFixed(2)}%`
        );
      }
    }
  });

  console.log("Assignment-1 Completed");
}

run();
