# mini-batch数据加载
import math
import random
import numpy as np

class DataLoader:
    # shuffle - 每个epoch当中是否打乱
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    # 从头开始循环
    def reset(self):
        self.iteration = 0
        if self.shuffle: # 打乱
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i+1) * batch_size]
        batch  = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])
        # print('x=', x, 'x.shape=', x.shape, 't=', t, "t.shape=", t.shape)
        # print('x.ndim=', x.ndim, "t.ndim=", t.ndim)

        self.iteration += 1
        return x, t # x.ndim=2, t.ndim=1

    def next(self):
        return self.__next__()

