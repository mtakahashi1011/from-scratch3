import numpy as np 
import sys
import os 

path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(path)

from dezero import *
from dezero.utils import *

def f(x):
    y = x**4 - 2 * x**2
    return y 

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)
    gx = x.grad 
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data