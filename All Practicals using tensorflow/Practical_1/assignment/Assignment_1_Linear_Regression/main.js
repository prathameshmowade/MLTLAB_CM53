const xs = tf.tensor1d([1, 2, 3, 4, 5])
const ys = tf.tensor1d([3, 6, 9, 12, 15])

const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'meanSquaredError'
})

async function runAssignment1() {
  console.log("Training Assignment 1 model...")
  await model.fit(xs, ys, { epochs: 300 })

  console.log("Actual Values:")
  ys.print()

  console.log("Predicted Values:")
  model.predict(xs).print()
}

runAssignment1()
