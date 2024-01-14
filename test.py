import pickle
import matplotlib.pyplot as plt

with open('history/cifar10_cnn_3layer_2_3_atanh.txt', 'rb') as f:
    accuracy = pickle.load(f)['accuracy']
plt.plot(accuracy)
plt.show()
