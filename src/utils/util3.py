import numpy as np

# 複数の入出力に対応できるようにVariableクラスを修正(ステップ11，12，13)
# cleargrad()メソッドの実装(ステップ14s)
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data 
        self.grad = None 
        self.creator = None 

    def set_creator(self, func):
        self.creator = func 

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # x, y = f.input, f.output 
            # x.grad = f.backward(y.grad)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                # 同じ変数を繰り返し使う場合に正しく計算できるように修正
                # x.grad = gx 
                if x.grad is None:
                    x.grad = gx 
                else:
                    x.grad = x.grad + gx 
                if x.creator is not None:
                    funcs.append(x.creator) 
    def cleargrad(self):
        self.grad = None

# 複数の入出力に対応できるようにFunctionクラスを修正(ステップ11，12，13)
class Function:
    # def __call__(self, input):
    #     x = input.data 
    #     y = self.forward(x)
    #     output = Variable(as_array(y))
    #     output.set_creator(self)
    #     self.input = input 
    #     self.output = output 
    #     return output 

    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs 
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1 
        return y

    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        y = x ** 2 
        return y 
    
    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0].data  
        gx = 2 * x * gy 
        return gx 

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data 
        gx = np.exp(x) * gy 
        return gx 

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x