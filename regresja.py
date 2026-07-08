import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('data/dane7.txt')

x = a[:, [0]]
y = a[:, [1]]

# podzielic na test i train, np.random.shuffle, 80% train, 20% test
np.random.shuffle(a)
split = int(0.8 * len(a))
train = a[:split]
test = a[split:]

x_train = train[:, [0]]
y_train = train[:, [1]]
x_test = test[:, [0]]
y_test = test[:, [1]]

# Model 1 : Cubic Polynomial Regression - wielomian stopnia 3
# [x^3, x^2, x, 1] @ [a, b, c, d] = y
c = np.hstack([x_train**3, x_train**2, x_train, np.ones(x_train.shape)])
v = np.linalg.inv(c.T @ c) @ c.T @ y_train

# Model 2 : Reciprocal (Moore-Penrose Pseudoinverse)
c1 = np.hstack([1 / x_train, np.ones(x_train.shape)])
v1 = np.linalg.pinv(c1) @ y_train

# Model 3 : Linear Regression
c2 = np.hstack([x_train, np.ones(x_train.shape)])
v2 = np.linalg.pinv(c2) @ y_train


# Helper - mean square error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Model 1: Cubic predictions
y_train_pred1 = v[0]*x_train**3 + v[1]*x_train**2 + v[2]*x_train + v[3]
y_test_pred1  = v[0]*x_test**3  + v[1]*x_test**2  + v[2]*x_test  + v[3]

# Model 2: Reciprocal predictions
y_train_pred2 = v1[0]/x_train + v1[1]
y_test_pred2  = v1[0]/x_test  + v1[1]

# Model 3: Linear predictions
y_train_pred3 = v2[0]*x_train + v2[1]
y_test_pred3  = v2[0]*x_test  + v2[1]

# Print errors
print(f"{'Model':<15} {'Train MSE':>12} {'Test MSE':>12} {'Ratio T/Tr':>12}")
print("-" * 55)
for name, tr, te in [
    ("Cubic",      mse(y_train, y_train_pred1), mse(y_test, y_test_pred1)),
    ("Reciprocal", mse(y_train, y_train_pred2), mse(y_test, y_test_pred2)),
    ("Linear",     mse(y_train, y_train_pred3), mse(y_test, y_test_pred3)),
]:
    ratio = te / tr
    print(f"{name:<15} {tr:>12.4f} {te:>12.4f} {ratio:>12.2f}")


# Smooth x range for plotting curves
xx = np.linspace(min(x), max(x), 150)  # dla gładszego wykresu

plt.scatter(x, y, color='red', marker='o', label='Dane', zorder=5)

# Model 1 - Cubic
plt.plot(xx, v[0]*xx**3 + v[1]*xx**2 + v[2]*xx + v[3],
         color='blue', label='Cubic')

# Model 2 - Reciprocal
plt.plot(xx, v1[0]/xx + v1[1],
         color='orange', label='Reciprocal')

# Model 3 - Linear
plt.plot(xx, v2[0]*xx + v2[1],
         color='green', label='Linear')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Models')
plt.show()
