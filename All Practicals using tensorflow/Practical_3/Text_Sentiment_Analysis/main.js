let model;

const vocabulary={
good:1,great:2,excellent:3,amazing:4,love:5,happy:6,
bad:7,terrible:8,awful:9,hate:10,worst:11,sad:12
};

function encodeText(text){

const words=text.toLowerCase().split(" ");
const encoded=[];

words.forEach(word=>{
encoded.push(vocabulary[word]||0);
});

while(encoded.length<5){
encoded.push(0);
}

return encoded.slice(0,5);
}

async function createModel(){

model=tf.sequential();

model.add(tf.layers.embedding({
inputDim:20,
outputDim:8,
inputLength:5
}));

model.add(tf.layers.lstm({units:16}));

model.add(tf.layers.dense({
units:1,
activation:"sigmoid"
}));

model.compile({
optimizer:"adam",
loss:"binaryCrossentropy",
metrics:["accuracy"]
});

const trainingData=tf.tensor2d([
encodeText("good"),
encodeText("great"),
encodeText("excellent"),
encodeText("love this"),
encodeText("amazing product"),
encodeText("bad"),
encodeText("terrible"),
encodeText("hate this"),
encodeText("worst product"),
encodeText("awful")
]);

const labels=tf.tensor2d([
[1],[1],[1],[1],[1],
[0],[0],[0],[0],[0]
]);

console.log("Training RNN...");

await model.fit(trainingData,labels,{epochs:50});

console.log("Model Ready");

}

createModel();

const chat=document.getElementById("chat");
const loader=document.getElementById("loader");
const input=document.getElementById("textInput");

let chart;

function addMessage(text,type){

const msg=document.createElement("div");
msg.className="msg "+type;
msg.innerText=text;

chat.appendChild(msg);
chat.scrollTop=chat.scrollHeight;

}

input.addEventListener("keypress",function(e){

if(e.key==="Enter"){

const text=input.value;
if(text==="") return;

addMessage(text,"user");

console.log("User:",text);

input.value="";

loader.style.display="block";

setTimeout(()=>{

const encoded=encodeText(text);
const tensor=tf.tensor2d([encoded]);

const prediction=model.predict(tensor);

const score=prediction.dataSync()[0];

let sentiment;

if(score>0.5){
sentiment="Positive 😊";
}else{
sentiment="Negative 😞";
}

console.log("Score:",score);
console.log("Sentiment:",sentiment);

loader.style.display="none";

addMessage("Sentiment: "+sentiment,"ai");

updateChart(score);

},600);

}

});

function updateChart(score){

const ctx=document.getElementById("chart");

if(!chart){

chart=new Chart(ctx,{
type:"bar",
data:{
labels:["Sentiment Score"],
datasets:[{
label:"Confidence",
data:[score],
backgroundColor:["#00ffb3"]
}]
},
options:{
scales:{
y:{min:0,max:1}
}
}
});

}else{

chart.data.datasets[0].data=[score];
chart.update();

}

}
