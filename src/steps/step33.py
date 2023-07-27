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
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)