import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data()  # num_words=vocab_size)
# print('X ==========')
# print(x_train.shape)
# print(type(x_train))
# print(x_train[0])
# print(type(x_train[0]))
# print(x_train[0][0])
# print('Y ==========')
# print(y_train.shape)
# print(type(y_train))
# print(y_train[0])
# print(type(y_train[0]))


ds_data = np.load('dataset/data.npy', allow_pickle=True)
ds_labels = np.load('dataset/labels.npy', allow_pickle=True)
print(ds_data.shape)
print(ds_labels.shape)
x_train, x_val, y_train, y_val = train_test_split(ds_data, ds_labels, train_size=0.01)
np.save(f'dataset_2/data', x_train)
np.save(f'dataset_2/labels', y_train)