import unittest
import numpy as np 
from util2 import * 

class SquareTest(unittest.TestCase):
    def test_backpropagation(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        y.grad = np.array(1.0)
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)
        print(x.grad)
        expected = np.array(3.29744254)
        self.assertEqual(round(x.grad - expected, 4), 0)

    def test_creator(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        self.assertEqual(y.creator, C)
        self.assertEqual(y.creator.input, b)
        self.assertEqual(y.creator.input.creator, B)
        self.assertEqual(y.creator.input.creator.input, a)

    def test_backpropagation2(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        y.grad = np.array(1.0)
        y.backward()
        print(x.grad)
        expected = np.array(3.29744254)
        self.assertEqual(round(x.grad - expected, 4), 0)

unittest.main()