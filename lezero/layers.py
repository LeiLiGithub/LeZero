# 封装隐藏层，内部由激活函数构成

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import weakref
import numpy as np
from lezero.core import Parameter
import lezero.functions as F


class Layer:
    def __init__(self):
        self._params = set() # 保存所有的params

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)): # Parameter、Layer 进行记录
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,) # 保证outputs是tuple
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0] # 返回tuple或者单个元素

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self): 
        for name in self._params: # yield是生成器，可用于for循环、next方法，只能被遍历一次
            obj = self.__dict__[name] 

            if isinstance(obj, Layer):
                yield from obj.params() #yield from获取生成器中的值
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
        
    # 生成 String-Parameter摊平集合
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = (parent_key + '/' + name) if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e: # 保存失败（如Ctrl+C）则删除
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items(): # key-String, param-Variable
            param.data = npz[key]

# 线性关系
class Linear(Layer):
    
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # 未指定in_size则延后处理
            self._init_W()

        if nobias: # 无偏置
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b') # size=输出层

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # 在传播时根据输入矩阵shape初始化权重
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b) # 调用functions.linear
        return y