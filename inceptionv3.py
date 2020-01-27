import tensorflow as tf
from tensorflow.keras.preprocessing import image
keras = tf.keras
import numpy as np
from sklearn.svm import SVC
import numpy.random as nprandom
from tensorflow.keras.models import Model

layerName = 'mixed5'
print('getting features from layer: ' + layerName)
#import the inception model, add a global average layer to get a feature vector of reasonable size
base_model = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False)
#base_model = tf.keras.applications.ResNet152V2(weights='imagenet',include_top=False)

model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
model = tf.keras.Sequential([model, global_average_layer])

print('model ready')

dataset = '2828RGB'
n_train = 6999
n_test = 3015

if (dataset == '2828L'):
    cols = 785
    dim1 = 28
    dim2 = 28
    dim3 = 1
    datastr = r'C:\Users\jordy\Documents\Python_nn_stuff\hmnist_28_28_L.csv'
if (dataset == '2828RGB'):
    cols = 2353
    dim1 = 28
    dim2 = 28
    dim3 = 3
    datastr = r'C:/Users/Diego/Documents/School/PatternSetMining/data/hmnist_28_28_RGB.csv'

if (dataset == '88L'):
    cols = 65
    dim1 = 8
    dim2 = 8
    dim3 = 1
    datastr = r'C:\Users\jordy\Documents\Python_nn_stuff\hmnist_8_8_L.csv'
if (dataset == '88RGB'):
    cols = 193
    dim1 = 8
    dim2 = 8
    dim3 = 3
    datastr = r'C:\Users\jordy\Documents\Python_nn_stuff\hmnist_8_8_RGB.csv'

tekst2 = np.loadtxt(datastr, delimiter=',',dtype=str, usecols=range(cols))
a = np.delete(tekst2,0,0) #delete header
q = a.astype(int)

#intialize arrays for storing images
x_data = np.empty([len(q),(cols - 1)])
y_data = np.empty([len(q),1])

#fill the x and y values (y=labels)
for i,n in enumerate(q):
    y = np.array(n[len(n)-1])
    x = np.array(n[:len(n)-1])
    x_data[i] = x
    y_data[i] = y

#reshape the arrays to approriate format for input to the neural network
x_data = np.expand_dims(x_data,axis=2)
x_data = x_data.reshape(len(x_data),dim1,dim2,dim3)
y_data = y_data.reshape(len(y_data),)

#inception requires minimum of 75,75 as input dimensions so resize to a multiple of 28, 28*4
x_data_112p = np.empty([len(x_data),37632])#37632 hardcoded resulting from the total values 112*112*3
x_data_112p = np.expand_dims(x_data_112p,axis=2)
x_data_112p = x_data_112p.reshape(len(x_data),112,112,dim3)

#first transform to an image, than resize to a higher resolution, and transform back to an array
for i,n in enumerate(x_data):
    img = image.array_to_img(n)
    img = img.resize((28*4,28*4))
    x_data_112p[i] = image.img_to_array(img)

#get the output of the final global pooling layer and use it as the feature vectors
features = model.predict(x_data_112p)

#randomize the indices
nprandom.seed(0)
nprandom.shuffle(features)
nprandom.seed(0)
nprandom.shuffle(y_data)

unique, counts = np.unique(y_data, return_counts=True)
d = dict(zip(unique, counts))
for k,v in d.items():#create class weights
	d[k] = 10015/(7*v)

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

print("training model on hmnist, feature vector shape: ")
print(features.shape)

#train the svm
clf = SVC(class_weight='balanced', kernel='linear', C=0.001)
clf.fit(features[:6999],y_data[:6999])
y_pred = clf.predict(features[7000:])
print("results for hmnist dataset")
print(balanced_accuracy_score(y_data[7000:], y_pred))
print(accuracy_score(y_data[7000:], y_pred))

tekst2 = np.loadtxt(r'C:/Users/Diego/Documents/School/PatternSetMining/data/ISIC_28_28_RGB.csv', delimiter=';',dtype=str, usecols=range(cols))
a = np.delete(tekst2,0,0) #delete header
q = a.astype(float).astype(int)

#intialize arrays for storing images
x_data = np.empty([len(q),(cols - 1)])
y_data = np.empty([len(q),1])

#fill the x and y values (y=labels)
for i,n in enumerate(q):
    y = np.array(n[len(n)-1])
    x = np.array(n[:len(n)-1])
    x_data[i] = x
    y_data[i] = y

#reshape the arrays to approriate format for input to the neural network
x_data = np.expand_dims(x_data,axis=2)
x_data = x_data.reshape(len(x_data),dim1,dim2,dim3)
y_data = y_data.reshape(len(y_data),)

#inception requires minimum of 75,75 as input dimensions so resize to a multiple of 28, 28*4
x_data_112p = np.empty([len(x_data),37632])#37632 hardcoded resulting from the total values 112*112*3
x_data_112p = np.expand_dims(x_data_112p,axis=2)
x_data_112p = x_data_112p.reshape(len(x_data),112,112,dim3)

#first transform to an image, than resize to a higher resolution, and transform back to an array
for i,n in enumerate(x_data):
    img = image.array_to_img(n)
    img = img.resize((28*4,28*4))
    x_data_112p[i] = image.img_to_array(img)

#get the output of the final global pooling layer and use it as the feature vectors
features = model.predict(x_data_112p)

#randomize the indices
nprandom.seed(0)
nprandom.shuffle(features)
nprandom.seed(0)
nprandom.shuffle(y_data)

y_pred = clf.predict(features)
print("resuls for ISIC dataset:")
print(balanced_accuracy_score(y_data, y_pred))
print(accuracy_score(y_data, y_pred))
