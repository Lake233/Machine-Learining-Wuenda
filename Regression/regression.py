import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.eye(5)
print(A)

path = '/Regression/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, cols-1:cols]
print(X.head())
print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(X.shape)
print(theta.shape)
print(y.shape)

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')
plt.show()

path = '/Regression/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2)

def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

final_theta2 = normalEqn(X, y)
print(final_theta2)