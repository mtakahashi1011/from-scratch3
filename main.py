import numpy as np 
from dezero import Variable, Model, MLP
import dezero.functions as F 
import dezero.layers as L 
from dezero import optimizers

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

model = MLP((10, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)
model.plot(x, to_file='graph_img/two_layer_net2.png')

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i %1000 == 0:
        print('iters: ', i, 'loss: ', loss.data)