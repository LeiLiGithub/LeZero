if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lezero
import lezero.datasets as D
import lezero.functions as F
from lezero import MLP, DataLoader, optimizers
import numpy as np

# def f(x):
#     x = x.flatten()
#     x = x.astype(np.float32)
#     x /= 255.0
#     return x

# train_set = D.MNIST(train=True, transform=f)
# test_set = D.MNIST(train=False, transform=f)

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = D.MNIST(train=True)
test_set = D.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        labelT = t
        y = model(x)
        # print('y', y.shape, '=', y)

        t = np.eye(10)[t] # one-hot
        # print('t', t.shape, '=', t)

        loss = F.softmax_with_loss(y, t)
        # print("loss=", loss)

        acc = F.accuracy(y, labelT)
        # print('acc=', acc)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch + 1))
    print('train loss： {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with lezero.no_grad():
        for x, t in test_loader:
            labelT = t
            y = model(x)
            t = np.eye(10)[t] # one-hot
            loss = F.softmax_with_loss(y, t)
            acc = F.accuracy(y, labelT)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(test_set)))