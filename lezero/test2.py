if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lezero.datasets as D
import numpy as np
import struct

# import matplotlib.pyplot as plt

train_set = D.MNIST(train=True, transform=None)
test_set = D.MNIST(train=False, transform=None)

# 保存png
print('save MNIST as png file...')

data0 = train_set.data[0]
print("data0.shape", data0.shape)

data00 = data0[0]
print("data00.shape", data00.shape)
print(data00.dtype)

dir_name = os.path.dirname(__file__)
print("dir_name", dir_name)

file_path = os.path.join(dir_name, "save.bin")
data00.tofile(file_path)
# print(data00)


dir_name = os.path.dirname(__file__)
print("dir_name", dir_name)
file_path = os.path.join(dir_name, "save.bin")
bin_file = np.fromfile(file_path)
# print(bin_file)


# print(len(train_set))
# print(len(test_set))

# x, t = train_set[0]
# print(type(x), x.shape)
# print(t)
# plt.imshow(x.reshape(28,28), cmap='gray')

# plt.axis('off')
# plt.show()

# for i, j in enumerate((100,10)):
#     print(i, j)
# y = np.array([0,0.01,0.02,3,0.04,5,6,7,8,9]).reshape(2,5)
# print(y)
# print(y.shape[0])
# t = np.array([3]).reshape(1,1)
# print(t)
# print(y[np.arange(2), t])

# c = np.where(y==t)[1]
# print('c', c)

# a=np.array([[ 0.,  1.,  0.],
#             [ 1.,  0.,  0.],
#             [ 0.,  0.,  1.]])
# b = np.where(a==1)[1]
# print(b)

# print('=======')
# t = np.zeros((10, y.shape[0]))
# print(t)
# s = np.array([3,4,5])
# t = np.eye(10)[s]
# print(t)