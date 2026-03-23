let model;

async function trainModel() {
    const status = document.getElementById("status");

    status.innerText = "⏳ Training model... Please wait";

    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor([1, 2, 3, 4]);
    const ys = tf.tensor([2, 4, 6, 8]);

    await model.fit(xs, ys, { epochs: 200 });

    await model.save('localstorage://my-model');

    status.innerText = "✅ Model trained & saved successfully!";
}