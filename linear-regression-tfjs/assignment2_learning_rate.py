import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

X = np.linspace(0, 10, 50)
Y = 3 * X + 5 + np.random.randn(50)
n = len(X)
learning_rates = [0.001, 0.01, 0.1]
epochs = 1000
for lr in learning_rates:
    m = 0
    b = 0

    for i in range(epochs):
        Y_pred = m * X + b
        dm = (-2/n) * sum(X * (Y - Y_pred))
        db = (-2/n) * sum(Y - Y_pred)
        m = m - lr * dm
        b = b - lr * db

    plt.plot(X, m * X + b, label="Learning Rate = " + str(lr))
plt.scatter(X, Y, label="Actual Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
