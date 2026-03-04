console.log("Assignment 3 – Tensor Math & Reduction Operations");

const tensor = tf.tensor1d([1, 4, 9, 16]);

console.log("Original Tensor:");
tensor.print();

console.log("Square of Tensor:");
tf.square(tensor).print();

console.log("Square Root of Tensor:");
tf.sqrt(tensor).print();

console.log("Sum of Elements:");
tf.sum(tensor).print();

console.log("Mean of Elements:");
tf.mean(tensor).print();

console.log("Maximum Value:");
tf.max(tensor).print();

console.log("Tensor Shape:");
console.log(tensor.shape);

console.log("Type Casting to float32:");
tensor.asType('float32').print();
