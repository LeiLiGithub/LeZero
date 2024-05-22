# CNN模型

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from lezero import MLP, DataLoader, optimizers, np, no_grad
from lezero.models import SimpleConvNet
from lezero import datasets as D
from lezero import functions as F

# 将数据以文本形式打印在控制台
def print_data_in_console():
    input_data, label = load_one_data(1024)
    print("input_data.shape=", input_data.shape, 'label=', label)
    for i in range(0,28):
        for j in range(0,28):
            has_dot = '*' if input_data[0][i*28+j] > 0 else ' '
            print(has_dot, end="")
        print("")

# 从train_set中读取单条数据&标签，并格式化为(1,784)大小
def load_one_data(data_idx):
    train_set = D.MNIST(train=True)
    # 0-数据，1-标签
    input_data = train_set.data[data_idx][0].reshape(1, -1)
    return input_data, train_set.label[data_idx]
    # input_data = train_set.data[data_idx][0]
    # print("input_data:", input_data.shape)
    # for i in input_data:
    #     for j in i:
    #         has_dot = '*' if j > 0 else ' '
    #         print(has_dot, end="")
    #     print("")

    
def run_train_infer():
    print('run_train_infer...')

    model_file = os.path.join(os.path.dirname(__file__), '..', 'cnn_v1.npz')
    
    skip_train = 0

    max_epoch = 1
    batch_size = 50
    # hidden_size = 1000

    train_set = D.MNIST(transform=None, train=True)
    test_set = D.MNIST(transform=None, train=False)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = SimpleConvNet()
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
                # print('accuracy=', acc)

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
            with no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = F.softmax_with_loss(y, t)
                    acc = F.accuracy(y, t)
                    sum_loss += float(loss.data) * len(t)
                    sum_acc += float(acc.data) * len(t)

            print('test loss: {:.4f}, accuracy: {:.4f}'.format(
                sum_loss / len(test_set), sum_acc / len(test_set)))

        # 保存模型
        model.save_weights(model_file)


    # # 完成训练，检测自定义输入识别效果
    # # 模拟使用train_set首条数据
    # input_idx = 149
    # input_data = train_set.data[input_idx][0].reshape(1, -1)
    # print("input_data=", input_data.shape)
    # # 使用模型推理

    # infer = model(input_data)
    # print('===================')
    # print('input_idx', input_idx)
    # print('inferencing...')
    # # print(type(infer))
    # print("infer.argmax()=", infer.data[0].argmax(axis=0))

    # # 打印llabel，验证
    # label_data = train_set.label[input_idx]
    # print('label=', label_data)

    print("Fin!!!")
    

# print_data_in_console()
# load_one_data(1)

run_train_infer()