const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const objectText = document.getElementById("object");
const fingerText = document.getElementById("fingers");

let model;


// CAMERA

async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({
video:true
});

video.srcObject = stream;

}

startCamera();


// LOAD OBJECT MODEL

async function loadModel(){

model = await cocoSsd.load();
console.log("Object model loaded");

}

loadModel();


// VIDEO READY

video.addEventListener("loadeddata",()=>{

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

detectFrame();

});


// OBJECT DETECTION

async function detectObjects(){

if(!model) return;

const predictions = await model.detect(video);

predictions.forEach(pred => {

if(pred.score > 0.6){

const [x,y,w,h] = pred.bbox;

ctx.strokeStyle="lime";
ctx.lineWidth=3;
ctx.strokeRect(x,y,w,h);

ctx.fillStyle="lime";
ctx.font="18px Arial";

ctx.fillText(pred.class,x,y-10);

objectText.innerText = pred.class;

}

});

}


// HAND TRACKING

const hands = new Hands({
locateFile:(file)=>{
return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}
});

hands.setOptions({

maxNumHands:2,
modelComplexity:1,
minDetectionConfidence:0.7,
minTrackingConfidence:0.7

});

hands.onResults(results=>{

if(results.multiHandLandmarks){

results.multiHandLandmarks.forEach(landmarks=>{

drawConnectors(ctx, landmarks, HAND_CONNECTIONS,{color:"#00FFFF",lineWidth:2});
drawLandmarks(ctx, landmarks,{color:"#FF0000",lineWidth:1});

countFingers(landmarks);

});

}

});


// FINGER COUNT

function countFingers(l){

let fingers=0;

if(l[8].y < l[6].y) fingers++;
if(l[12].y < l[10].y) fingers++;
if(l[16].y < l[14].y) fingers++;
if(l[20].y < l[18].y) fingers++;

fingerText.innerText = fingers;

}


// MAIN LOOP

async function detectFrame(){

ctx.drawImage(video,0,0,canvas.width,canvas.height);

await detectObjects();

await hands.send({image:video});

requestAnimationFrame(detectFrame);

}