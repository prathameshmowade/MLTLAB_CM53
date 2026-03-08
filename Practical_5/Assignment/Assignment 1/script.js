const video = document.getElementById("video");
const result = document.getElementById("result");
const confidence = document.getElementById("confidence");

let model;


// Start Webcam

async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({
video:true
});

video.srcObject = stream;

}


// Load MobileNet Model

async function loadModel(){

model = await mobilenet.load();

console.log("MobileNet Model Loaded");

result.innerText = "Model Ready";

}


// Classify Objects

async function classifyFrame(){

if(model){

const predictions = await model.classify(video);

const top = predictions[0];

result.innerText = top.className;

confidence.innerText =
"Confidence: " + (top.probability*100).toFixed(2) + "%";

}

requestAnimationFrame(classifyFrame);

}


startCamera();
loadModel();

video.addEventListener("loadeddata", classifyFrame);