import unittest 
import numpy as np 

# OK
# from dezero.core_simple import Variable
#from dezero import Variable 
from dezero import *
from dezero.utils import _dot_var, _dot_func, plot_dot_graph
import dezero.functions as F

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

unittest.main()