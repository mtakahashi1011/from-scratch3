import numpy as np 
import os 
import sys 

path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(path)

from dezero import Variable, Model
import dezero.functions as F 
import dezero.layers as L 

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# model = L.Layer()
# model.l1 = L.Linear(10)
# model.l2 = L.Linear(1)

# def predict(x):
#     y = model.l1(x)
#     y = F.sigmoid_simple(y)
#     y = model.l2(y)
#     return y 

model = TwoLayerNet(10, 1)
model.plot(x, to_file='graph_img/two_layer_net2.png')

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data 
    if i %1000 == 0:
        print('iters: ', i, 'loss: ', loss.data)