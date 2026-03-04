console.log("TensorFlow.js Version:");
console.log(tf.version.tfjs);

// Scalar
console.log("Scalar:");
const scalar = tf.scalar(5);
scalar.print();

// Vector
console.log("Vector:");
const vector = tf.tensor1d([1, 2, 3]);
vector.print();

// Matrix
console.log("Matrix:");
const matrix = tf.tensor2d([[1, 2], [3, 4]]);
matrix.print();

// Addition
console.log("Added Vector:");
const addedVector = tf.add(vector, tf.tensor1d([4, 5, 6]));
addedVector.print();
