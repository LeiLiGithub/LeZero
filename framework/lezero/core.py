is_android = False
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import weakref
import contextlib
if not is_android:
    import lezero

########################
######## 核心类 ########
########################


# 全局配置
class Config:
    enable_backprop = True # 是否允许反向传播，训练时开启，推理时关闭


# 最基本的运算单元
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # 计算前序全部梯度，采用循环方式
    # retain_grad: 是否保留梯度，梯度用于反向传播，仅推理时不需要保留梯度
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x : x.generation)

        if self.creator is not None: # skip the first variable
            add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs] # output是weakref

            with using_config('enable_backprop', create_graph): # 只有允许二次反向传播时才计算

                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx


                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # weakref，释放以使引用计数归零

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # 重置导数
    def cleargrad(self):
        self.grad = None

    # 调整形状，支持多种输入
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return lezero.functions.reshape(self, shape)

    # 转置
    def transpose(self):
        return lezero.functions.transpose(self)

    # 对全部元素求和
    def sum(self, axis=None, keepdims=False):
        return lezero.functions.sum(self, axis, keepdims)

    # 转置
    @property
    def T(self):
        return lezero.functions.transpose(self)

    def __len__(self):
        return len(self.data)

    # toString
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        else:
            p = str(self.data).replace('\n', '\n' + ' ' * 9)
            return 'variable(' + p + ')'

# 所有函数的基类，核心方法forward、backward
class Function:
    def __call__(self, *inputs): # *将变长参数转换为列表
        inputs = [as_variable(x) for x in inputs] # 支持直接输入ndarray
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 使用*解包，ys是ndarray类型
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: # 训练时开启反向传播，推理时关闭
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] # 弱引用防止循环引用
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

# 用作权重参数
class Parameter(Variable):
    pass

########################
########运算符重载########
########################

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = lezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = lezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

class Pow(Function):
    def __init__(self, c): # x是变量，c是常量
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

################################
######## 核心类用到的函数 ########
################################

# 可以通过 with 调用
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value) # 还原

# 简便写法，不需反向传播
def no_grad():
    return using_config('enable_backprop', False)

# 参数统一化
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 包装成Variable
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1): # x0是self，x1是被减数（左项）
    x1 = as_array(x1)
    return Sub()(x1, x0)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def neg(x):
    return Neg()(x)

def pow(x, c):
    return Pow(c)(x)

# Variable运算符重载
def setup_variable():
    Variable.__add__ = add
    Variable.__mul__ = mul
    Variable.__radd__ = add
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow