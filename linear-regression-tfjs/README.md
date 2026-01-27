
# Linear Regression using TensorFlow.js (CDN)

🔗 **Project Repository Link:**
[https://github.com/prathameshmowade/MLTLAB_CM53/tree/main/linear-regression-tfjs](https://github.com/prathameshmowade/MLTLAB_CM53/tree/main/linear-regression-tfjs)

---

## Aim

To implement a **simple Linear Regression model** using **TensorFlow.js** with
synthetic data and execute it directly in the **web browser** using CDN.

---

## Objectives

* To understand **client-side machine learning**
* To create a regression model using `tf.sequential()`
* To use `tf.layers.dense()` for prediction
* To train and test the model inside the browser
* To observe model output using the browser console

---

## Tools & Technologies Used

* HTML
* JavaScript
* TensorFlow.js (CDN)
* VS Code
* Web Browser (Chrome / Edge)

---

## Project Structure

```
linear-regression-tfjs/
│
├── index.html
├── script.js
└── README.md
```

---

## Description

This project demonstrates a **basic Linear Regression model** trained on
synthetic (x, y) data using **TensorFlow.js**. The entire model creation,
training, and prediction process runs on the **client side** without using
any server or backend.

TensorFlow.js is loaded using a **CDN**, making the setup simple and lightweight.

---

## Working Principle

1. TensorFlow.js library is loaded via CDN in `index.html`
2. Synthetic training data is created in `script.js`
3. A Sequential model is defined with one Dense layer
4. The model is compiled using Mean Squared Error loss
5. Training is performed using `.fit()`
6. Predictions are checked using the browser console

---

## How to Run the Project

1. Open the `index.html` file in any modern web browser
2. Right-click → **Inspect**
3. Open the **Console** tab
4. Train the model by typing:

   ```javascript
   trainModel()
   ```
5. Test prediction using:

   ```javascript
   model.predict(tf.tensor2d([10], [1, 1])).print()
   ```

---

## Output

The model predicts output values for given input based on the learned linear
relationship from the synthetic dataset.

Example output:

```
Tensor
[[11.8]]
```

---

## Advantages

* No backend or server required
* Runs completely in the browser
* Easy to understand for beginners
* Fast setup using CDN

---

## Applications

* Learning basic Machine Learning concepts
* Understanding regression models
* Client-side ML applications
* Educational and lab experiments

---

## Conclusion

This experiment successfully demonstrates how a **Linear Regression model**
can be implemented and trained entirely on the **client side** using
TensorFlow.js. It helps beginners understand the fundamentals of regression
and browser-based machine learning.

---

## Author

**Prathamesh Mowade**
SB Jain Institute of Technology, Management & Research

---

## License

This project is for **educational and academic use only**.

---


