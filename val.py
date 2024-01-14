import glob
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from train_myself_model import get_mnist_dataset, get_fashion_mnist_dataset, get_cifar10_dataset

import tensorflow as tf
from tensorflow.keras.models import load_model

all_net_info = {}
all_net_info['Network'] = []
all_net_info['Accuracy'] = []
all_net_info['Loss'] = []
all_net_info['Epochs'] = []
all_net_path = [net_path.replace('\\', '/') for net_path in glob.glob('models/*.h5')]
for net_path in all_net_path:
    model = load_model(net_path)
    net_name = net_path.split('/')[-1].split('.')[0]
    dataset = net_name.split('_')[0]
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    loss, accuracy = model.evaluate(x_test, y_test)

    with open('history/{}.txt'.format(net_name), 'rb') as f:
        epochs = len(pickle.load(f)['loss'])

    all_net_info['Network'].append(net_name)
    all_net_info['Accuracy'].append(accuracy)
    all_net_info['Loss'].append(loss)
    all_net_info['Epochs'].append(epochs)

df = pd.DataFrame(all_net_info)
df.to_csv('test1.csv')

