import unittest
import numpy as np 
from util3 import * 

class Test(unittest.TestCase):
    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        print(y.data)
        self.assertEqual(y.data, 5)

    def test_square(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(z.data, 13)
        self.assertEqual(x.grad, 4)
        self.assertEqual(y.grad, 6)

    def test_use_same_variable(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, np.array(2.0))

    def test_clear_grad(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, np.array(2.0))

        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, np.array(3.0))

unittest.main()