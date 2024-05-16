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


def run_train_infer():

    model_file = 'mlp_v1.npz'
    skip_train = 0

    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = D.MNIST(train=True)
    test_set = D.MNIST(train=False)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 10))
    optimizer = optimizers.SGD().setup(model)

    if os.path.exists(model_file):
        model.load_weights(model_file)
        skip_train = 1

    if skip_train == 0:
        for epoch in range(max_epoch):
            sum_loss, sum_acc = 0, 0

            # 训练
            for x, t in train_loader: # t是label
                # labelT = t
                y = model(x)
                # print('y', y.shape, '=', y)

                # t = np.eye(10)[t] # one-hot
                # print('t', t.shape, '=', t)

                loss = F.softmax_with_loss(y, t)
                # print("loss=", loss)

                acc = F.accuracy(y, t)
                # print('acc=', acc)

                model.cleargrads()
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

            print('epoch: {}'.format(epoch + 1))
            print('train loss： {:.4f}, accuracy: {:.4f}'.format(
                sum_loss / len(train_set), sum_acc / len(train_set)))

            # 推理
            sum_loss, sum_acc = 0, 0
            with lezero.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = F.softmax_with_loss(y, t)
                    acc = F.accuracy(y, t)
                    sum_loss += float(loss.data) * len(t)
                    sum_acc += float(acc.data) * len(t)

            print('test loss: {:.4f}, accuracy: {:.4f}'.format(
                sum_loss / len(test_set), sum_acc / len(test_set)))

        model.save_weights(model_file)


    # 完成训练，检测自定义输入识别效果
    # 模拟使用train_set首条数据
    input_idx = 149
    input_data = train_set.data[input_idx][0].reshape(1, -1)
    print("input_data=", input_data.shape)
    # 使用模型推理

    infer = model(input_data)
    print('===================')
    print('input_idx', input_idx)
    print('inferencing...')
    # print(type(infer))
    print("infer.argmax()=", infer.data[0].argmax(axis=0))

    # 打印llabel，验证
    label_data = train_set.label[input_idx]
    print('label=', label_data)

run_train_infer()