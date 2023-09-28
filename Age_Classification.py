
import numpy as np 
import pandas as pd 
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical


os.chdir('D:\\6th sem\\Gender & Age classification\\utkface_aligned_cropped\\UTKFace')
im =Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128,128))
# im

onlyfiles = os.listdir()
print(len(onlyfiles))

shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]

classes = []
for i in age:
    i = int(i)
    if i <= 14:
        classes.append(0)
    if (i>14) and (i<=25):
        classes.append(1)
    if (i>25) and (i<40):
        classes.append(2)
    if (i>=40) and (i<60):
        classes.append(3)
    if i>=60:
        classes.append(4)


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face =cv2.resize(face, (32, 32) )
    X_data.append(face)


X = np.squeeze(X_data)
print(X.shape)

# normalize data
X = X.astype('float32')
X /= 255

classes[:10]

categorical_labels = to_categorical(classes, num_classes=5)

categorical_labels[:10]

(x_train, y_train), (x_test, y_test) = (X[:15008],categorical_labels[:15008]) , (X[15008:] , categorical_labels[15008:])
(x_valid , y_valid) = (x_test[:7000], y_test[:7000])
(x_test, y_test) = (x_test[7000:], y_test[7000:])

len(x_train)+len(x_test) + len(x_valid) == len(X)



model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))


# Take a look at the model summary
model.summary()


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=20,
         validation_data=(x_valid, y_valid),)

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


labels =["CHILD",  # index 0
        "YOUTH",      # index 1
        "ADULT",     # index 2 
        "MIDDLEAGE",        # index 3 
        "OLD",         # index 4
        ]

y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth

# figure = plt.figure(figsize=(20, 8))
# for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
#     ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
#     # Display each image
#     ax.imshow(np.squeeze(x_test[index]))
#     predict_index = np.argmax(y_hat[index])
#     true_index = np.argmax(y_test[index])
#     # Set the title for each image
#     ax.set_title("{} ({})".format(labels[predict_index], 
#                                   labels[true_index]),
#                                   color=("green" if predict_index == true_index else "red"))
# plt.show()

model.save('D:\\6th sem\\Gender & Age classification\\Age_Classification_model_ep20.h5')


from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model=load_model('D:\\6th sem\\Gender & Age classification\\Age_Classification_model_ep20.h5')

# model.evaluate(x_train,y_train)


import cv2
import numpy as np

image=cv2.imshow('sumit',x_test[0])
cv2.waitKey(0)

img_path=input("Enter the image path: ")
img=cv2.imread(img_path)
cv2.imshow("sumit",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=np.array(img)
img=cv2.resize(img,(32,32))
img.shape
img=np.expand_dims(img,0)
output=model.predict(img)
final_out=np.argmax(output)
print("The given image is: ",labels[final_out])

