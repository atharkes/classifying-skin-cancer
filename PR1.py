import numpy as np
import os

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.utils import to_categorical

dataset = '2828RGB'
n_train = 6999
n_test = 3015
binary_labels = True

n_labels = 2 if binary_labels else 7
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
dims = (dim1, dim2, dim3)
txt = np.loadtxt(data_path, delimiter=',', dtype=str, usecols=range(cols))
a = np.delete(txt, 0, 0)
q = a.astype(int)
np.random.shuffle(q)

train = q[0:n_train]
x_train = np.empty([n_train, (cols - 1)])
y_train = np.empty([n_train, 1])
for i, row in enumerate(train):
    y = int(np.array(row[len(row)-1]))
    x = np.array(row[:len(row)-1])
    x_train[i] = x
    if not binary_labels:
        # Using 7 labels
        y_train[i] = y
    if binary_labels and y != 0 and y != 1 and y != 6:
        # Benign for Binary labels
        y_train[i] = 0
    if binary_labels and y != 2 and y != 3 and y != 4 and y != 5:
        # Malignant for Binary labels
        y_train[i] = 1

test = q[(n_train + 1):]
train_new = np.expand_dims(x_train, axis=2)
x_train2 = train_new.reshape((n_train, dim1, dim2, dim3))
y_train2 = y_train.reshape(n_train, )

x_test = np.empty([n_test, (cols - 1)])
y_test = np.empty([n_test, 1])
for i, n in enumerate(test):
    y = np.array(n[len(n)-1])
    x = np.array(n[:len(n)-1])
    x_test[i] = x
    if not binary_labels:  # when using 7 labels
        y_test[i] = y
    if binary_labels and y != 0 and y != 1 and y != 6:  # benign binary labels
        y_test[i] = 0
    if binary_labels and y != 2 and y != 3 and y != 4 and y != 5:  # malignant binary labels
        y_test[i] = 1
test_new = np.expand_dims(x_test, axis=2)
x_test2 = test_new.reshape((n_test, dim1, dim2, dim3))
y_test2 = y_test.reshape(n_test, )

num_filters = 10
filter_size = 2
pool_size = 2
padding = 'same'

# Build the model.
model = Sequential()
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Activation('relu'))
model.add(Conv2D(num_filters, filter_size, input_shape=dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(n_labels, activation='softmax'))

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model
model.fit(
    x_train2,
    to_categorical(y_train2),
    epochs=200,
    shuffle=True,
    batch_size=50,
    validation_data=(x_test2, to_categorical(y_test2)),
)

acc = model.evaluate(x_test2, to_categorical(y_test2))
dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
model_path = "{0}\\Models\\model{1}_{2:.2f}_{3:.2f}_{4}.h5".format(file_path, dataset, acc[1], acc[0], dt_string)
model.save(model_path)
