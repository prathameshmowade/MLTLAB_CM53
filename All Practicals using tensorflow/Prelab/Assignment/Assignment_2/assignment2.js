console.log("Assignment 2 – Basic Tensor Operations");

const A = tf.tensor2d([[1, 2], [3, 4]]);
const B = tf.tensor2d([[5, 6], [7, 8]]);

console.log("Matrix A:");
A.print();

console.log("Matrix B:");
B.print();

console.log("Matrix Addition (A + B):");
tf.add(A, B).print();

console.log("Matrix Multiplication (A x B):");
tf.matMul(A, B).print();

console.log("Transpose of Matrix A:");
tf.transpose(A).print();

console.log("Accessing Tensor Data (arraySync):");
console.log(A.arraySync());
