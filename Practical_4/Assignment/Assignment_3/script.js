let mobileModel;
let cocoModel;
let modelsLoaded = false;

// Load Models
async function loadModels() {

    mobileModel = await mobilenet.load();
    cocoModel = await cocoSsd.load();

    modelsLoaded = true;

    document.getElementById("loading").innerText = "Models Loaded Successfully!";
    console.log("Both models loaded successfully");
}

loadModels();

// Change Image
function changeImage() {
    const selected = document.getElementById("imageSelect").value;
    document.getElementById("preview").src = "images/" + selected;
}

// Compare Models
async function compareModels() {

    if (!modelsLoaded) {
        alert("Models are still loading...");
        return;
    }

    const img = document.getElementById("preview");

    const mobilePredictions = await mobileModel.classify(img);
    const cocoPredictions = await cocoModel.detect(img);

    displayMobile(mobilePredictions);
    displayCoco(cocoPredictions);
}

// Display MobileNet
function displayMobile(predictions) {

    let output = "";

    for (let i = 0; i < 3; i++) {

        output += `
            <div class="prediction">
                <strong>${i+1}. ${predictions[i].className}</strong>
                <br>
                Confidence: ${(predictions[i].probability * 100).toFixed(2)}%
            </div>
        `;
    }

    document.getElementById("mobileResult").innerHTML = output;
}

// Display COCO
function displayCoco(predictions) {

    let output = "";

    if (predictions.length === 0) {
        output = "<div class='prediction'>No objects detected</div>";
    } else {

        for (let i = 0; i < predictions.length && i < 3; i++) {

            output += `
                <div class="prediction">
                    <strong>${i+1}. ${predictions[i].class}</strong>
                    <br>
                    Confidence: ${(predictions[i].score * 100).toFixed(2)}%
                </div>
            `;
        }
    }

    document.getElementById("cocoResult").innerHTML = output;
}