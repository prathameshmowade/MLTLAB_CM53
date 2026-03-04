let model;
let modelLoaded = false;

let totalTests = 0;
let correctPredictions = 0;

async function loadModel() {
    model = await mobilenet.load();
    modelLoaded = true;
    document.getElementById("loading").innerText = "Model Loaded Successfully!";
}

loadModel();

function changeImage() {
    const selected = document.getElementById("imageSelect").value;
    document.getElementById("preview").src = "images/" + selected;
}

async function classifyImage() {

    if (!modelLoaded) {
        alert("Model is still loading...");
        return;
    }

    const img = document.getElementById("preview");
    const selectedImage = document.getElementById("imageSelect").value.replace(".jpg", "");

    const predictions = await model.classify(img);

    let output = "";
    for (let i = 0; i < 3; i++) {
        output += `
        <div class="prediction">
            ${i+1}. ${predictions[i].className}
            <br>Confidence: ${(predictions[i].probability * 100).toFixed(2)}%
        </div>`;
    }

    document.getElementById("result").innerHTML = output;

    totalTests++;

    if (predictions[0].className.toLowerCase().includes(selectedImage)) {
        correctPredictions++;
    }

    let accuracy = ((correctPredictions / totalTests) * 100).toFixed(2);

    document.getElementById("accuracyBox").innerText =
        "Tests: " + totalTests +
        " | Correct: " + correctPredictions +
        " | Accuracy: " + accuracy + "%";
}