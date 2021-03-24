import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# vocab_size = 20000  # Only consider the top 20k words
# num_tokens_per_example = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data()  # num_words=vocab_size)


print('X ==========')
print(x_train.shape)
print(type(x_train))

print(x_train[0])
print(type(x_train[0]))

print(x_train[0][0])

print('Y ==========')
print(y_train.shape)
print(type(y_train))

print(y_train[0])
print(type(y_train[0]))
