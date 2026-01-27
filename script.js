async function trainModel() {

    // Step 1: Create synthetic data
    const x = tf.tensor2d([1, 2, 3, 4, 5], [5, 1])
    const y = tf.tensor2d([2, 4, 6, 8, 10], [5, 1])

    // Step 2: Create model
    const model = tf.sequential()

    // Step 3: Add dense layer
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }))

    // Step 4: Compile model
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    })

    // Step 5: Train model
    await model.fit(x, y, {
        epochs: 200
    })

    // Step 6: Predict new value
    const output = model.predict(tf.tensor2d([6], [1, 1]))

    output.print()
}

trainModel()
