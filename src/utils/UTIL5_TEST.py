import unittest
import numpy as np 
from util5 import * 

class Test(unittest.TestCase):
    def test_overload(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = a * b + c 
        y.backward()

        self.assertEqual(y.data, np.array(7.0))
        self.assertEqual(a.grad, np.array(2.0))
        self.assertEqual(b.grad, np.array(3.0))
    
    def test_as_variable(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        self.assertEqual(y.data, np.array(5.0))
    
    def test_overload2(self):
        x = Variable(np.array(2.0))
        y = 3.0 * x + 1.0
        self.assertEqual(y.data, np.array(7.0))

    def test_overload3(self):
        x = Variable(np.array(2.0))
        y = np.array(1.0) + np.array(3.0) * x
        self.assertEqual(y.data, np.array(7.0))

    def test_overload4(self):
        x = Variable(np.array(2.0))
        y = - x 
        self.assertEqual(y.data, np.array(-2.0))
        z = 5.0 - x
        self.assertEqual(z.data, np.array(3.0))

    def test_overload5(self):
        x = Variable(np.array(6.0))
        y = 24.0 / x 
        self.assertEqual(y.data, np.array(4.0))
        z = x / 2.0 
        self.assertEqual(z.data, np.array(3.0))

    def test_overload6(self):
        x = Variable(np.array(3.0))
        y = x ** 2 
        self.assertEqual(y.data, np.array(9.0))

unittest.main()