let model;
let modelLoaded = false;

async function loadModel() {
    model = await mobilenet.load();
    modelLoaded = true;
    document.getElementById("loading").innerText = "Model Loaded Successfully!";
    console.log("MobileNet Loaded");
}

loadModel();

async function classifyImage() {

    if (!modelLoaded) {
        alert("Model is still loading. Please wait...");
        return;
    }

    const img = document.getElementById("preview");

    const predictions = await model.classify(img);

    let output = "";

    for (let i = 0; i < 3; i++) {
        output += `
        <div class="prediction">
            <strong>${i+1}. ${predictions[i].className}</strong><br>
            Confidence: ${(predictions[i].probability * 100).toFixed(2)}%
        </div>
        `;
    }

    document.getElementById("result").innerHTML = output;
}