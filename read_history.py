import glob
import pickle
import matplotlib.pyplot as plt

all_txt_path = [txt_path.replace('\\', '/') for txt_path in glob.glob('history/*.txt')]
for txt_path in all_txt_path:
    txt_name = txt_path.split('/')[-1]
    with open(txt_path, 'rb') as f:
        history = pickle.load(f)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.title(txt_name)
    plt.show()

# with open('history/fashion_mnist_fnn_3x50_atanh.txt', 'rb') as f:
#     history = pickle.load(f)
# plt.plot(history['loss'], label='loss')
# plt.plot(history['val_loss'], label='val_loss')
# plt.legend(['train', 'val'], loc='upper right')
# plt.show()