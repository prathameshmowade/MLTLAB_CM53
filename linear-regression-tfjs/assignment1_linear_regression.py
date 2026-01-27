import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

X = np.linspace(0, 10, 50)
Y = 4 * X + 3 + np.random.randn(50)
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    Y_pred = m * X + b

    dm = (-2/n) * sum(X * (Y - Y_pred))
    db = (-2/n) * sum(Y - Y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, m * X + b, color="red", label="Predicted Line")
plt.xlabel("Input X")
plt.ylabel("Output Y")
plt.legend()
plt.show()
