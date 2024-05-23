# 封装模型对象，模型内部由多个隐藏层构成

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lezero.layers as L
import lezero.functions as F
from lezero import utils
import numpy as np
from lezero.layers import Convolution, Pooling, Relu, Affine, Sigmoid

from collections import OrderedDict

class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# Multi-Layer Perceptron 多层感知器
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x)) # 依次执行Linear+激活函数
        return self.layers[-1](x) # 最后一层


# CNN: Conv-Relu-Pool Affine-Relu Affine-Softmax
# 类似MLP
class SimpleConvNet(Model):
    def __init__(
        self,
        input_dim=(1,28,28), # C W H
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01
        ):
        super().__init__() # 初始化layer中的_params
        # 卷积层（过滤器）池化层 初始化
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # 卷积层输出，长、宽相等
        conv_output_size = (input_size + 2*filter_pad - filter_size) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 初始化权重
        self.params_dict = {}
        # Conv
        self.params_dict['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params_dict['b1'] = np.zeros(filter_num)
        # Affine
        self.params_dict['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params_dict['b2'] = np.zeros(hidden_size)
        # Affine
        self.params_dict['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params_dict['b3'] = np.zeros(output_size)

        # 生成层
        # Conv-ReLU-Pooling-Affine-ReLU-Affine-Softmax
        self.layers = []

        self.Conv1 = Convolution(self.params_dict['W1'], self.params_dict['b1'], conv_param['stride'], conv_param['pad'])
        self.layers.append(self.Conv1)

        self.Relu1 = Relu()
        self.layers.append(self.Relu1)

        self.Pool1 = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers.append(self.Pool1)

        self.Affine1 = Affine(self.params_dict['W2'], self.params_dict['b2'])
        self.layers.append(self.Affine1)

        self.Relu2 = Relu()
        self.layers.append(self.Relu2)

        self.Affine2 = Affine(self.params_dict['W3'], self.params_dict['b3'])
        self.layers.append(self.Affine2)

        # self.last_layer = SoftmaxWithLoss()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x
