import unittest 
import numpy as np 

from dezero import *
from dezero.datasets import Spiral
from dezero.utils import _dot_var, _dot_func, plot_dot_graph
import dezero.functions as F
import dezero.layers as L

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
        z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) \
            * (30 + (2*x - 3*y)**2 * (18 -32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
        z.backward()
        x.name = 'x'
        y.name = 'y'
        z.name = 'z'
        # plot_dot_graph(z, False, to_file='graph_img/goldstein.png')

    def test_second_derivative(self):
        def f(x):
            y = x**4 - 2 * x**2
            return y 

        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        print(x.grad)
        self.assertEqual(x.grad.data, np.array(24.0))

        gx = x.grad
        x.cleargrad()
        gx.backward()
        print(x.grad)
        self.assertEqual(x.grad.data, np.array(44.0))

    def test_newton_method(self):
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

        self.assertEqual(x.data, np.array(1.0))

    def test_sin(self):
        x = Variable(np.array(1.0))
        y = F.sin(x)
        y.backward(create_graph=True)

        exact_values = [-0.841470984, -0.540302305, 0.841470984]

        for i in range(3):
            gx = x.grad 
            x.cleargrad()
            gx.backward(create_graph=True)
            print(x.grad.data)
            # print(round(x.grad.data-exact_values[i], 6))
            self.assertEqual(round(x.grad.data-exact_values[i], 6), 0)

    def test_tanh(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = 'x'
        y.name = 'y'
        y.backward(create_graph=True)

        iters = 0

        for i in range(iters):
            gx = x.grad 
            x.cleargrad()
            gx.backward(create_graph=True)
        
        gx = x.grad 
        gx.name = 'gx' + str(iters+1)
        # plot_dot_graph(gx, verbose=False, to_file='graph_img/tanh.png')

    def test_reshape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)
        print(x.grad.data)
        exact_value = np.array([[1, 1, 1], [1, 1, 1]])
        self.assertEqual((x.grad.data == exact_value).all(), True)

    def test_reshape_method(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y1 = x.reshape((3, 2))
        y2 = x.reshape(3, 2)

    def test_transpose(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        y.backward(retain_grad=True)
        print(x.grad.data)
        exact_value = np.array([[1, 1, 1], [1, 1, 1]])
        self.assertEqual((x.grad.data == exact_value).all(), True)

    def test_sum_to_in_backprop(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1 
        exact_y = np.array([11, 12, 13])
        self.assertEqual((y.data == exact_y).all(), True)
        y.backward(retain_grad=True)
        print('y.grad', y.grad)
        self.assertEqual(x1.grad.data, np.array(3))

    def test_sum(self):
        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = F.sum(x)
        y.backward()
        print('y', y.data)
        self.assertEqual(y.data, np.array(21))
        print('x.grad', x.grad.data)
        self.assertEqual((x.grad.data == np.array([1, 1, 1, 1, 1, 1])).all(), True)

    def test_sum2(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x, axis=0)
        y.backward()
        self.assertEqual((y.data == np.array([5, 7, 9])).all(), True)
        self.assertEqual((x.grad.data == np.array([[1, 1, 1], [1, 1, 1]])).all(), True)
    
    def test_sum3(self):
        x = Variable(np.random.randn(2 ,3, 4, 5))
        y = F.sum(x, keepdims=True)
        self.assertEqual((y.shape == (1, 1, 1, 1)), True)

    def test_matmul(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = F.matmul(x, W)
        y.backward()
        self.assertEqual((x.grad.shape==(2, 3)), True)
        self.assertEqual((W.grad.shape==(3 ,4)), True)

    def test_layer(self):
        layer = Layer()

        layer.p1 = Parameter(np.array(1))
        layer.p2 = Parameter(np.array(2))
        layer.p3 = Variable(np.array(3))
        layer.p4 = 'test'

        print(layer._params)
        print('---------------')

        for name in layer._params:
            print(name, layer.__dict__[name])

    def test_cross_entropy(self):
        x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
        t = np.array([2, 0, 1, 0])
        model = MLP((10, 3))
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        print('cross entropy loss', loss)

    def test_spiral(self):
        train_set = Spiral(train=True)
        print('train_set', train_set[0])
        self.assertEqual(len(train_set), 300)

unittest.main()