let model;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let detectionRunning = false;

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// 🔹 Load COCO-SSD
async function loadModel() {
    document.getElementById("status").innerText = "Loading Model...";
    model = await cocoSsd.load();
    document.getElementById("status").innerText = "Model Ready ✅";
}

// 🔹 Save model (demo purpose)
async function saveModel() {
    document.getElementById("status").innerText = "Saving Model...";

    // COCO-SSD cannot be directly saved, so we simulate using tf model
    const tempModel = tf.sequential();
    tempModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    await tempModel.save('localstorage://demo-model');

    document.getElementById("status").innerText = "Model Saved in Browser ✅";
}

// 🔹 Load saved model
async function loadSavedModel() {
    document.getElementById("status").innerText = "Loading Saved Model...";

    await tf.loadLayersModel('localstorage://demo-model');

    document.getElementById("status").innerText = "Saved Model Loaded ✅";
}

// 🔹 Start detection manually
function runPrediction() {
    if (!model) {
        document.getElementById("status").innerText = "Load model first!";
        return;
    }

    detectionRunning = true;
    detectFrame();
}

// 🔹 Detection loop
async function detectFrame() {
    if (!detectionRunning) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const predictions = await model.detect(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let counts = {};
    let listHTML = "";

    predictions.forEach(pred => {
        const [x, y, w, h] = pred.bbox;

        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "#00FF00";
        ctx.fillText(
            `${pred.class} (${(pred.score * 100).toFixed(1)}%)`,
            x,
            y > 10 ? y - 5 : 10
        );

        counts[pred.class] = (counts[pred.class] || 0) + 1;
        listHTML += `<li>${pred.class}</li>`;
    });

    document.getElementById("objectList").innerHTML = listHTML;

    let countText = "";
    for (let key in counts) {
        countText += `${key}: ${counts[key]}<br>`;
    }

    document.getElementById("counts").innerHTML = countText;

    requestAnimationFrame(detectFrame);
}

// 🔹 Start everything
async function start() {
    await setupCamera();
    await loadModel();
}

start();