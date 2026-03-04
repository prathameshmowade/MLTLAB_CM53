async function run() {
  const xs = tf.tensor2d([1,2,3,4,5], [5,1])
  const ys = tf.tensor2d([3,5,7,9,11], [5,1])

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  })

  await model.fit(xs, ys, { epochs: 150 })

  const unseen = tf.tensor2d([12, 15], [2,1])
  console.log("Predictions for unseen data:")
  model.predict(unseen).print()
}

run()
