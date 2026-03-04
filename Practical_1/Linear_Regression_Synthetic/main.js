const xs = tf.tensor1d([1, 2, 3, 4, 5])
const ys = tf.tensor1d([2, 4, 6, 8, 10])

const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'meanSquaredError'
})

async function trainModel() {
  console.log("Training started...")
  await model.fit(xs, ys, { epochs: 200 })
  console.log("Training completed")

  console.log("Prediction for x = 6")
  model.predict(tf.tensor1d([6])).print()
}

trainModel()
