console.log("TensorFlow.js version:", tf.version.tfjs);

const t1 = tf.tensor([1, 2, 3]);
const t2 = tf.tensor([4, 5, 6]);

t1.add(t2).print();
