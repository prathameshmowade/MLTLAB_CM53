let model;

async function loadModel() {
    document.getElementById("result").innerText = "Loading model...";

    model = await tf.loadLayersModel('localstorage://my-model');

    document.getElementById("result").innerText = "Model loaded successfully!";
}

function predict() {
    const value = document.getElementById("inputValue").value;

    if (!model) {
        document.getElementById("result").innerText = "Load model first!";
        return;
    }

    const input = tf.tensor([Number(value)]);
    const output = model.predict(input);
    const prediction = output.dataSync()[0];

    // Expected output (y = 2x)
    const expected = value * 2;

    document.getElementById("result").innerText =
        `Prediction: ${prediction.toFixed(2)} | Expected: ${expected}`;
}