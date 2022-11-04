import numpy as np
import matplotlib.pyplot as plt
import math


# This function generate a toy dataset for regression
def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def MSE_Ridge(X, y, a, b, alpha=0.0):
    n = len(y)
    loss = 0
    for i in range(n):
        y_predicted = a*X[i] + b
        loss = loss + (1/n) * (y[i] - y_predicted)**2
    # regularization :
    loss = loss + alpha*(a**2)/n
    return loss


def gradient_descent(X, y, a, b, alpha, lr=0.01):
    a_gradient = 0
    b_gradient = 0
    n = len(y)
    for i in range(n):
        a_gradient += ((-2/n) * (y[i] - (a*X[i] + b)) * X[i]) + (2*alpha*a)/n
        b_gradient += ((-2/n) * (y[i] - (a*X[i] + b))) + (2*alpha*b)/n
    # norm of the gradient
    grad = math.sqrt(a_gradient**2 + b_gradient**2)
    # update the parameters a and b
    a = a - lr*a_gradient
    b = b - lr*b_gradient
    return a, b, grad


def cross_validation(X, y, a, b, kfolds=5):  # find the best regularization parameter
    n = len(y)
    start = 0
    end = int(n/kfolds)-1
    X_fold1 = X[start:end]
    y_fold1 = y[start:end]
    start = end + 1
    end += int(n/kfolds)
    X_fold2 = X[start:end]
    y_fold2 = y[start:end]
    start = end + 1
    end += int(n/kfolds)
    X_fold3 = X[start:end]
    y_fold3 = y[start:end]
    start = end + 1
    end += int(n/kfolds)
    X_fold4 = X[start:end]
    y_fold4 = y[start:end]
    start = end + 1
    end += int(n/kfolds)
    X_fold5 = X[start:end]
    y_fold5 = y[start:end]

    train_data = list()
    train_targets = list()
    test_data = list()
    test_targets = list()

    train_data1 = X_fold1 + X_fold2 + X_fold3 + X_fold4
    train_targets1 = y_fold1 + y_fold2 + y_fold3 + y_fold4
    test_data.append(X_fold5)
    test_targets.append(y_fold5)
    train_data.append(train_data1)
    train_targets.append(train_targets1)

    train_data2 = X_fold1 + X_fold2 + X_fold3 + X_fold5
    train_targets2 = y_fold1 + y_fold2 + y_fold3 + y_fold5
    test_data.append(X_fold4)
    test_targets.append(y_fold4)
    train_data.append(train_data2)
    train_targets.append(train_targets2)

    train_data3 = X_fold1 + X_fold2 + X_fold3 + X_fold5
    train_targets3 = y_fold1 + y_fold2 + y_fold3 + y_fold5
    test_data.append(X_fold3)
    test_targets.append(y_fold3)
    train_data.append(train_data3)
    train_targets.append(train_targets3)

    train_data4 = X_fold1 + X_fold2 + X_fold3 + X_fold5
    train_targets4 = y_fold1 + y_fold2 + y_fold3 + y_fold5
    test_data.append(X_fold2)
    test_targets.append(y_fold2)
    train_data.append(train_data4)
    train_targets.append(train_targets4)

    train_data5 = X_fold2 + X_fold3 + X_fold4 + X_fold5
    train_targets5 = y_fold2 + y_fold3 + y_fold4 + y_fold5
    test_data.append(X_fold1)
    test_targets.append(y_fold1)
    train_data.append(train_data5)
    train_targets.append(train_targets5)

    min_loss = float('inf')
    best_alpha = 0

    for i_alpha in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
        for k in range(kfolds):
            # training
            maxIter = 1000
            for i in range(maxIter):
                # # Loss function :
                # loss = MSE_Ridge(train_data[k], train_targets[k], a, b, alpha)

                # Update a and b using gradient descent :
                a, b, grad = gradient_descent(train_data[k], train_targets[k], a, b, alpha=i_alpha)

                if abs(grad) < eps:  # if the slope is approximately zero
                    break

            # testing
            loss = MSE_Ridge(test_data[k], test_targets[k], a, b, alpha=i_alpha)
            if loss < min_loss:
                min_loss = loss
                best_alpha = i_alpha

    print(f'min loss : {min_loss}')
    print(f'best alpha : {best_alpha}')
    return best_alpha


X, y = make_wave(n_samples=50)
a, b = 0, 0
lr = 0.01  # the learning rate
eps = 1e-6  # the accuracy
maxIter = 1000  # maximum number of iteration
best_alpha = cross_validation(X, y, a, b)  # the regularization parameter

print(best_alpha)

for i in range(maxIter):
    # Loss function :
    loss = MSE_Ridge(X, y, a, b, alpha=best_alpha)
    print('loss : ', loss)
    # Update a and b using gradient descent :
    a, b, grad = gradient_descent(X, y, a, b, alpha=best_alpha, lr=0.01)
    if abs(grad) < eps:  # if the slope is approximately zero
        print(f'grad : {grad}')
        break

print('\n')
print(f'a, b : {a}, {b}')
print(f'Alpha : {best_alpha}')
estimated_function = a*X + b
plt.scatter(X, y, label='Training data')
plt.title('Ridge regression')
plt.plot(X, estimated_function, '-r', label='Estimated function')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

