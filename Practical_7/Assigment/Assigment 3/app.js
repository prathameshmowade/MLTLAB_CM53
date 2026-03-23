let model;

async function trainModel() {
    document.getElementById("result").innerText = "Training...";

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor([1, 2, 3, 4]);
    const ys = tf.tensor([2, 4, 6, 8]);

    await model.fit(xs, ys, { epochs: 200 });

    document.getElementById("result").innerText = "Model trained!";
}

// 🔹 Export model
async function saveModel() {
    await model.save('downloads://my-model');
    document.getElementById("result").innerText = "Model downloaded!";
}

// 🔹 Import model
async function loadModel() {
    const jsonFile = document.getElementById("jsonFile").files[0];
    const weightsFile = document.getElementById("weightsFile").files[0];

    model = await tf.loadLayersModel(
        tf.io.browserFiles([jsonFile, weightsFile])
    );

    document.getElementById("result").innerText = "Model loaded!";
}

// 🔹 Predict
function predict() {
    const value = document.getElementById("inputValue").value;

    const input = tf.tensor([Number(value)]);
    const output = model.predict(input);
    const prediction = output.dataSync()[0];

    const expected = value * 2;

    document.getElementById("result").innerText =
        `Prediction: ${prediction.toFixed(2)} | Expected: ${expected}`;
}