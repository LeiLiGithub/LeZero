if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lezero.datasets as D
import numpy as np
# import matplotlib.pyplot as plt

# train_set = D.MNIST(train=True, transform=None)
# test_set = D.MNIST(train=False, transform=None)

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
y = np.array([0,0.01,0.02,3,0.04,5,6,7,8,9]).reshape(2,5)
print(y)
print(y.shape[0])
t = np.array([3]).reshape(1,1)
print(t)
print(y[np.arange(2), t])

c = np.where(y==t)[1]
print('c', c)

# a=np.array([[ 0.,  1.,  0.],
#             [ 1.,  0.,  0.],
#             [ 0.,  0.,  1.]])
# b = np.where(a==1)[1]
# print(b)

print('=======')
t = np.zeros((10, y.shape[0]))
print(t)
s = np.array([3,4,5])
t = np.eye(10)[s]
print(t)