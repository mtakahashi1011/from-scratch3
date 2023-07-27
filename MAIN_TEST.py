import unittest 
import numpy as np 

# OK
# from dezero.core_simple import Variable
#from dezero import Variable 
from dezero import *
from dezero.utils import _dot_var, _dot_func

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

unittest.main()