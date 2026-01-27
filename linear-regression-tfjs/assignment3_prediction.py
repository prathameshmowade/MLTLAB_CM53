import numpy as np
X = np.array([1, 2, 3, 4, 5])
Y = np.array([8, 11, 14, 17, 20])
n = len(X)
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
for i in range(epochs):
    Y_pred = m * X + b
    dm = (-2/n) * sum(X * (Y - Y_pred))
    db = (-2/n) * sum(Y - Y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db
X_new = np.array([6, 7, 8])
Y_new = m * X_new + b
for x, y in zip(X_new, Y_new):
    print("Input:", x, "Predicted Output:", round(y, 2))
