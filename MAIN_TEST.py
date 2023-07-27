import unittest 
import numpy as np 

# OK
# from dezero.core_simple import Variable
#from dezero import Variable 
from dezero import *
from dezero.utils import _dot_var

class Dezero_Test(unittest.TestCase):
    def test_package(self):
        x = Variable(np.array(1.0))
        self.assertEqual(x.data, np.array(1.0))

    def test_dot_var(self):
        x = Variable(np.random.randn(2, 3))
        x.name = 'x'
        print(_dot_var(x))
        print(_dot_var(x, verbose=True))

unittest.main()