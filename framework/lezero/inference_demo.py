is_android = False

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lezero import datasets as D
from lezero import MLP
from lezero import np


# 打印用户输入
def print_user_input(int_array):
    input_data = np.array(int_array).reshape(1, -1)
    print("input_data.shape=", input_data.shape)
    for i in range(0,28):
        for j in range(0,28):
            has_dot = '*' if input_data[0][i*28+j] > 0 else ' '
            print(has_dot, end="")
        print("")

# 推理用户输入
def infer_user_input(user_input):
    # 用np.array将java.jarray('I')转为nparray
    input_data = np.array(user_input)

    model_file_name = 'mlp_v2.npz'
    model_file_path = os.path.join(os.path.dirname(__file__), '..', model_file_name)
    
    hidden_size = 1000
    model = MLP((hidden_size, 10))

    if os.path.exists(model_file_path):
        model.load_weights(model_file_path)
        print('load finish:', model_file_path)
    else:
        raise ValueError(model_file_path, 'not exist!')

    print(model(input_data).data)
    infer = model(input_data).data.argmax(axis=0)
    print("infer=", infer)
    return infer


def run_inference(input_idx):
    print(os.path.dirname(__file__))
    model_file_name = 'mlp_v2.npz'
    model_file_path = os.path.join(os.path.dirname(__file__), '..', model_file_name)
    
    hidden_size = 1000
    model = MLP((hidden_size, 10))

    if os.path.exists(model_file_path):
        model.load_weights(model_file_path)
        print('load finish:', model_file_path)
    else:
        raise ValueError(model_file_path, 'not exist!')

    train_set = D.MNIST(train=True)

    input_data = train_set.data[input_idx][0].reshape(1, -1)
    infer = model(input_data)

    print("infer=", infer.data[0].argmax(axis=0))

    # 打印label，验证
    label_data = train_set.label[input_idx]
    print('label=', label_data)

# if not is_android:
    # run_inference(415)