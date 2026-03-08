const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;


// Start webcam

async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({
video:true
});

video.srcObject = stream;

}


// Load COCO SSD model

async function loadModel(){

model = await cocoSsd.load();

console.log("Model Loaded");

detectObjects();

}


// Detect objects

async function detectObjects(){

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

const predictions = await model.detect(video);

ctx.clearRect(0,0,canvas.width,canvas.height);

predictions.forEach(pred => {

const [x,y,width,height] = pred.bbox;

ctx.strokeStyle = "cyan";
ctx.lineWidth = 3;

ctx.strokeRect(x,y,width,height);

ctx.fillStyle = "cyan";
ctx.font = "18px Arial";

ctx.fillText(
pred.class + " " + (pred.score*100).toFixed(1) + "%",
x,
y>10 ? y-5 : 10
);

});

requestAnimationFrame(detectObjects);

}


startCamera();

video.addEventListener("loadeddata", loadModel);