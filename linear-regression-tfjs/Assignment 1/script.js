async function run() {
  const x = [1,2,3,4,5,6,7,8,9,10]
  const y = x.map(v => 2 * v + 1)

  const xs = tf.tensor2d(x, [x.length, 1])
  const ys = tf.tensor2d(y, [y.length, 1])

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  })

  await model.fit(xs, ys, { epochs: 200 })

  console.log("Actual values:")
  ys.print()

  console.log("Predicted values:")
  model.predict(xs).print()
}

run()
