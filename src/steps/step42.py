import numpy as np 
import sys 
import os 

path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(path)

from dezero import Variable 
import dezero.functions as F 

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b 
    return y 

def mea_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = mea_squared_error(y, y_pred)
    
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data 
    b.data -= lr * b.grad.data 
    print(W.data, b.data, loss.data)