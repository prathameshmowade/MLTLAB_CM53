const xs = tf.tensor1d([1, 2, 3, 4, 5])
const ys = tf.tensor1d([2, 4, 6, 8, 10])

async function trainWithLearningRate(lr) {
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  model.compile({
    optimizer: tf.train.sgd(lr),
    loss: 'meanSquaredError'
  })

  console.log("Training with learning rate:", lr)
  await model.fit(xs, ys, { epochs: 120 })

  console.log("Prediction for x = 6")
  model.predict(tf.tensor1d([6])).print()
}

trainWithLearningRate(0.001)
trainWithLearningRate(0.01)
trainWithLearningRate(0.1)
