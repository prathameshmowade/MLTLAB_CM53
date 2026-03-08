const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fpsDisplay = document.getElementById("fps");

let model;

let lastTime = performance.now();
let frames = 0;


// Start webcam

async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({
video:true
});

video.srcObject = stream;

}


// Load model

async function loadModel(){

model = await cocoSsd.load();

console.log("Model Loaded");

detect();

}


// Detect objects

async function detect(){

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

const predictions = await model.detect(video);

ctx.clearRect(0,0,canvas.width,canvas.height);

predictions.forEach(pred=>{

const [x,y,width,height] = pred.bbox;

ctx.strokeStyle="cyan";
ctx.lineWidth=3;

ctx.strokeRect(x,y,width,height);

ctx.fillStyle="cyan";
ctx.font="18px Arial";

ctx.fillText(
pred.class + " " + (pred.score*100).toFixed(1)+"%",
x,
y>10 ? y-5 : 10
);

});

calculateFPS();

requestAnimationFrame(detect);

}


// FPS calculation

function calculateFPS(){

frames++;

const now = performance.now();

if(now-lastTime >= 1000){

fpsDisplay.innerText = "FPS: " + frames;

frames = 0;
lastTime = now;

}

}


startCamera();

video.addEventListener("loadeddata", loadModel);