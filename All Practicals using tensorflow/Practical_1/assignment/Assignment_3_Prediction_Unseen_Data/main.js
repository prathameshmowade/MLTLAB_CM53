const xs = tf.tensor1d([1, 2, 3, 4, 5])
const ys = tf.tensor1d([5, 10, 15, 20, 25])

const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'meanSquaredError'
})

async function predictUnseen() {
  console.log("Training model for Assignment 3...")
  await model.fit(xs, ys, { epochs: 250 })

  console.log("Prediction for unseen input x = 6")
  model.predict(tf.tensor1d([6])).print()

  console.log("Expected value ≈ 30")
}

predictUnseen()
