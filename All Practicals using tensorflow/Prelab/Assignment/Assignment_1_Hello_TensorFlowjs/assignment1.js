console.log("---- Tensor Creation ----");

const scalar = tf.scalar(10);
scalar.print();

const vector = tf.tensor1d([1, 2, 3]);
vector.print();

const matrix = tf.tensor2d([[1, 2], [3, 4]]);
matrix.print();

console.log("---- Element-wise Operations ----");

const v1 = tf.tensor1d([2, 4, 6]);
const v2 = tf.tensor1d([1, 3, 5]);

tf.add(v1, v2).print();
tf.mul(v1, v2).print();

console.log("---- Reshape vs Flatten ----");

const t = tf.tensor2d([[1, 2], [3, 4]]);
t.reshape([4, 1]).print();
t.flatten().print();
