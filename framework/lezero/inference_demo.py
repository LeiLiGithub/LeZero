is_android = False

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if is_android:
    import datasets as D
    from models import MLP
else:
    from lezero import datasets as D
    from lezero import MLP



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

if not is_android:
    run_inference(415)