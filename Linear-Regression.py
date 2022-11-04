import numpy as np
import matplotlib.pyplot as plt
import math


# This function generate a toy dataset for regression
def make_wave(n_samples=100):
    rnd = np.random.RandomState(5)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def MSError(X, y, a, b):
    loss = 0
    n = len(y)
    for i in range(n):
        y_predicted = a*X[i] + b
        loss += (y[i] - y_predicted)**2
    return (1/n)*loss


def gradient_descent(X, y, a, b, learning_rate=0.01):
    a_gradient = 0
    b_gradient = 0
    n = len(y)
    for i in range(n):
        a_gradient += -(2/n) * (y[i] - (a*X[i] + b)) * X[i]
        b_gradient += -(2/n) * (y[i] - (a*X[i] + b))
    # norm of the gradient
    grad = math.sqrt(a_gradient**2 + b_gradient**2)
    # update the parameters a and b :
    a = a - learning_rate*a_gradient
    b = b - learning_rate*b_gradient
    return a, b, grad


X, y = make_wave(n_samples=50)
n = len(y)
a = 0
b = 0
# f(x) = a*x + b
lr = 1e-2  # the learning rate
eps = 1e-6  # the accuracy
maxIter = 1000  # maximum number of iteration

for i in range(maxIter):
    # Loss function :
    loss = MSError(X, y, a, b)
    print('loss : ', loss)
    # Update a and b using gradient descent :
    a, b, grad = gradient_descent(X, y, a, b)
    if abs(grad) < eps:  # if the slope is approximately zero
        print(f'grad : {grad}')
        break

print('\n')
print(a, b)
estimated_function = a*X + b
plt.scatter(X, y, label='Training data')
plt.title('Linear regression')
plt.plot(X, estimated_function, '-r', label='Estimated function')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

