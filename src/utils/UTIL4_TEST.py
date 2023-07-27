import unittest
import numpy as np 
from util4 import * 

class Test(unittest.TestCase):
    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(x.grad, np.array(64.0))

    def test_retain_grad(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        self.assertEqual(y.grad, None)
        self.assertEqual(t.grad, None)
        self.assertEqual(x0.grad, np.array(2.0))
        self.assertEqual(x1.grad, np.array(1.0))

unittest.main()