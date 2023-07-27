import unittest 
import numpy as np 

# OK
# from dezero.core_simple import Variable
from dezero import Variable 

class Dezero_Test(unittest.TestCase):
    def test_package(self):
        x = Variable(np.array(1.0))
        self.assertEqual(x.data, np.array(1.0))

unittest.main()