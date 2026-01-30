async function run() {
  const xs = tf.tensor2d([1,2,3,4,5,6,7,8,9,10], [10,1])
  const ys = tf.tensor2d([3,5,7,9,11,13,15,17,19,21], [10,1])

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  model.compile({
    optimizer: tf.train.sgd(0.1), // change: 0.001 / 0.01 / 0.1
    loss: 'meanSquaredError'
  })

  await model.fit(xs, ys, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: Loss = ${logs.loss}`)
      }
    }
  })
}

run()
