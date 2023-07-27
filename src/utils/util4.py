import numpy as np
import weakref
import contextlib

# 世代の追加(ステップ16)
# 弱参照(参照カウントを増やすことなく別のオブジェクトを参照する機能)の追加(ステップ17)
# 微分を消去するモードの追加(ステップ18)
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data 
        self.grad = None 
        self.creator = None 
        self.generation = 0

    def set_creator(self, func):
        self.creator = func 
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        # funcs = [self.creator]
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx 
                else:
                    x.grad = x.grad + gx 
                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator) 

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None 
    def cleargrad(self):
        self.grad = None

# クラス変数としてenable_backupを導入(ステップ18)
class Config:
    enable_backup = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backup', False)

# 逆伝播の有効無効を切り替えるモードの実装(ステップ18)
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backup:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs]

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