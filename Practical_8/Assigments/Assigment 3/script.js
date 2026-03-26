let webcam;
let net;
let classifier;

let dataset = {};
let labels = [];
let currentClass = null;
let previousAccuracy = null;

// Setup Webcam
async function setupWebcam() {
const cam = document.getElementById('webcam');
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
cam.srcObject = stream;

return new Promise(resolve => {
cam.onloadedmetadata = () => resolve(cam);
});
}

// Load Model
async function loadModel() {
net = await mobilenet.load();
console.log("✅ MobileNet Loaded");
}

// Init
async function init() {
webcam = await setupWebcam();
await loadModel();
}
init();

// Add Class
function addNewClass() {
const name = document.getElementById("newClass").value.trim();
if (!name) return;

if (labels.includes(name)) {
alert("Class already exists!");
return;
}

labels.push(name);
dataset[name] = [];

const btn = document.createElement("button");
btn.innerText = name;
btn.onclick = () => setClass(name);

document.getElementById("classButtons").appendChild(btn);
document.getElementById("newClass").value = "";
}

// Select Class
function setClass(name) {
currentClass = name;
document.getElementById("results").innerText = "Selected: " + name;
}

// Capture Image
function capture() {
if (!currentClass) {
alert("Select class first!");
return;
}

if (!net) {
alert("Model loading...");
return;
}

const act = net.infer(webcam, true);

if (!act || !act.shape) {
alert("Invalid capture!");
return;
}

dataset[currentClass].push(act);
console.log("Captured:", currentClass);
}

// Train Model
async function train() {

if (labels.length < 2) {
alert("Add at least 2 classes!");
return;
}

let trainXs = [];
let trainYs = [];
let testXs = [];
let testYs = [];

for (let i = 0; i < labels.length; i++) {
const label = labels[i];
const data = dataset[label];


if (!Array.isArray(data) || data.length < 5) {
  alert("Add 5+ samples for " + label);
  return;
}

const split = Math.floor(data.length * 0.8);

for (let j = 0; j < data.length; j++) {
  const item = data[j];

  if (!item || !item.shape) continue;

  if (j < split) {
    trainXs.push(item);
    trainYs.push(i);
  } else {
    testXs.push(item);
    testYs.push(i);
  }
}


}

const validTrainXs = trainXs.filter(x => x && x.shape);

if (validTrainXs.length === 0) {
alert("Invalid training data!");
return;
}

const xsTrain = tf.concat(validTrainXs);
const ysTrain = tf.oneHot(tf.tensor1d(trainYs, 'int32'), labels.length);

classifier = tf.sequential();

classifier.add(tf.layers.dense({
inputShape: [xsTrain.shape[1]],
units: 100,
activation: 'relu'
}));

classifier.add(tf.layers.dense({
units: labels.length,
activation: 'softmax'
}));

classifier.compile({
optimizer: tf.train.adam(0.0001),
loss: 'categoricalCrossentropy'
});

document.getElementById("results").innerText = "Training... ⏳";

await classifier.fit(xsTrain, ysTrain, { epochs: 20 });

document.getElementById("results").innerText = "✅ Training Done!";
}

// Evaluate Model
async function evaluateModel() {

if (!classifier) {
alert("Train first!");
return;
}

let testXs = [];
let testYs = [];

for (let i = 0; i < labels.length; i++) {
const label = labels[i];
const data = dataset[label];


if (!Array.isArray(data) || data.length < 2) {
  alert("Not enough test data for " + label);
  return;
}

const split = Math.floor(data.length * 0.8);

for (let j = split; j < data.length; j++) {
  const item = data[j];

  if (!item || !item.shape) continue;

  testXs.push(item);
  testYs.push(i);
}


}

const validTestXs = testXs.filter(x => x && x.shape);

if (validTestXs.length === 0) {
alert("Invalid test data!");
return;
}

const xsTest = tf.concat(validTestXs);

let correct = 0;

for (let i = 0; i < validTestXs.length; i++) {
const pred = classifier.predict(xsTest.slice([i, 0], [1]));
const result = await pred.data();


const predicted = result.indexOf(Math.max(...result));

if (predicted === testYs[i]) correct++;


}

const acc = (correct / validTestXs.length) * 100;

let text = "Accuracy: " + acc.toFixed(2) + "%";

if (previousAccuracy !== null) {
text += "<br>Change: " + (acc - previousAccuracy).toFixed(2) + "%";
}

previousAccuracy = acc;

document.getElementById("results").innerHTML = text;
}
