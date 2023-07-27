import unittest 
import numpy as np 

# OK
# from dezero.core_simple import Variable
#from dezero import Variable 
from dezero import *
from dezero.utils import _dot_var, _dot_func, plot_dot_graph

class Dezero_Test(unittest.TestCase):
    def test_package(self):
        x = Variable(np.array(1.0))
        self.assertEqual(x.data, np.array(1.0))

    def test_dot_var(self):
        x = Variable(np.random.randn(2, 3))
        x.name = 'x'
        print(_dot_var(x))
        print(_dot_var(x, verbose=True))

    def test_dot_func(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y = x0 + x1 
        txt = _dot_func(y.creator)
        print(txt)
        
    def test_plot_dot_graph(self):
        x0 = Variable(np.array(1.0), 'x0')
        x1 = Variable(np.array(1.0), 'x1')
        y = x0 + x1 
        y.name = 'y'
        plot_dot_graph(y, False)

    def test_goldstein_price(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x - 3*y)**2 * (18 -32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
        z.backward()
        x.name = 'x'
        y.name = 'y'
        z.name = 'z'
        plot_dot_graph(z, False, to_file='graph_img/goldstein.png')
unittest.main()