import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical

dataset = '2828RGB'
n_train = 6999
n_test = 3015
file_path = os.path.dirname(os.path.realpath(__file__))
data_path = ''
cols = dim1 = dim2 = dim3 = 0

if dataset == '2828L':
    cols = 785
    dim1 = 28
    dim2 = 28
    dim3 = 1
    data_path = file_path + r'\Data\hmnist_28_28_L.csv'
if dataset == '2828RGB':
    cols = 2353
    dim1 = 28
    dim2 = 28
    dim3 = 3
    data_path = file_path + r'\Data\hmnist_28_28_RGB.csv'
if dataset == '88L':
    cols = 65
    dim1 = 8
    dim2 = 8
    dim3 = 1
    data_path = file_path + r'\Data\hmnist_8_8_L.csv'
if dataset == '88RGB':
    cols = 193
    dim1 = 8
    dim2 = 8
    dim3 = 3
    data_path = file_path + r'\Data\hmnist_8_8_RGB.csv'

txt = np.loadtxt(data_path, delimiter=',', dtype=str, usecols=range(cols))
a = np.delete(txt, 0, 0)
q = a.astype(int)

train = q[0:n_train]
x_train = np.empty([n_train, (cols - 1)])
y_train = np.empty([n_train, 1])
for i, n in enumerate(train):
    y = np.array(n[len(n) - 1])
    x = np.array(n[:len(n) - 1])
    x_train[i] = x
    y_train[i] = y
test = q[(n_train + 1):]
train_new = np.expand_dims(x_train, axis=2)
x_train2 = train_new.reshape(n_train, dim1, dim2, dim3)
y_train2 = y_train.reshape(n_train, )

x_test = np.empty([n_test, (cols - 1)])
y_test = np.empty([n_test, 1])
for i, n in enumerate(test):
    y = np.array(n[len(n) - 1])
    x = np.array(n[:len(n) - 1])
    x_test[i] = x
    y_test[i] = y
test_new = np.expand_dims(x_test, axis=2)
x_test2 = test_new.reshape(n_test, dim1, dim2, dim3)
y_test2 = y_test.reshape(n_test, )

# x_train2 = np.expand_dims(x_train2,axis=3)
# x_test2 = np.expand_dims(x_test2,axis=3)

num_filters = 10
filter_size = 5
pool_size = 2
padding = 'same'

# Build the model.
model = Sequential()
model.add(Conv2D(num_filters, filter_size, input_shape=(dim1, dim2, dim3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(num_filters, filter_size, input_shape=(dim1, dim2, dim3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

# Compile the model.
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model.
model.fit(
    x_train2,
    to_categorical(y_train2),
    epochs=200,
    shuffle=True,
    batch_size=20,
    validation_data=(x_test2, to_categorical(y_test2)),
)
