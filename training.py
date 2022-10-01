import imp
from itertools import count
from operator import le
from statistics import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import pickle

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


# Parameters
path = 'myData'
labels = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epoch_val = 10
imageDim = (32,32,3)
testRatio = 0.2
validationRatio = 0.2


# import images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print('total classes detected: ', len(myList))
noOfClasses = len(myList)
print('importing classes....... ')
for x in range (0, len(myList)):
    myPicturesList = os.listdir(path+'/'+str(count))
    for y in myPicturesList:
        curImg = cv2.imread(path+'/'+str(count)+'/'+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end="    ")
    count +=1
print('  ')

images = np.array(images)
classNo = np.array(classNo)


# split data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validationRatio)


# checking for errors
print('data shapes')
print('train: ', end='  ');print(x_train.shape, y_train.shape)
print('validation: ', end='  ');print(x_valid.shape, y_valid.shape)
print('test: ', end='  ');print(x_test.shape, y_test.shape)
assert(x_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(x_valid.shape[0]==y_valid.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(x_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in testing set"
assert(x_test.shape[1:]==(imageDim))," The dimesions of the testing images are wrong "
assert(x_valid.shape[1:]==(imageDim))," The dimesions of the validation images are wrong "
assert(x_train.shape[1:]==(imageDim))," The dimesions of the Training images are wrong "


# read csv file (labels)
data = pd.read_csv(labels)
print('data shape: ', data.shape, type(data))


# display images of each classs
no_Samples = []
cols = 5
no_Classes = noOfClasses
fig,axs = plt.subplots(nrows=no_Classes, ncols=cols, figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j,rows in data.iterrows():
        x_select = x_train[y_train == j]
        axs[j][i].imshow(x_select[random.randint(0, len(x_select)-1),:,:], cmap=plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j)+'-'+rows['Name'])
            no_Samples.append(len(x_select))


# display bar chart
print(no_Samples)
plt.figure(figsize=(12,4))
plt.bar(range(0, no_Classes), no_Samples)
plt.title('ditribution of training dataset')
plt.xlabel('class number')
plt.ylabel('number of images')
plt.show()


# processing the images
def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def processing(img):
    img = gray(img)
    img = equalize(img)
    img = img/255
    return img

x_train = np.array(list(map(processing,x_train)))
x_valid = np.array(list(map(processing,x_valid)))
x_test = np.array(list(map(processing,x_test)))
cv2.imshow('grayscale images', x_train[random.randint(0, len(x_train)-1), :, :])

# adding depth
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)


# augment images
dataGenAug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGenAug.fit(x_train)
batches = dataGenAug.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)


# show augmented images samples
fig,axs = plt.subplots(1,15)
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDim[0],imageDim[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_valid = to_categorical(y_valid, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# cnn
def myModel():
    noFilters = 60
    noNodes = 500
    sizeFilters = (5,5)
    sizeFilters2 = (3,3)
    sizePool = (2,2)
    
    model = Sequential()
    model.add(Conv2D(noFilters, sizeFilters, input_shape=(imageDim[0], imageDim[1],1), activation= 'relu'))
    model.add(Conv2D(noFilters, sizeFilters, activation= 'relu'))
    model.add(MaxPooling2D(pool_size=sizePool))

    model.add(Conv2D(noFilters//2, sizeFilters2, activation='relu'))
    model.add(Conv2D(noFilters//2, sizeFilters2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# train
model = myModel()
print(model.summary())
# history = model.fit_generator(dataGenAug.flow(x_train, y_train, batch_size= batch_size_val), steps_per_epoch= steps_per_epoch_val, epochs= epoch_val, validation_data=(x_valid, y_valid), shuffle=1)

history = model.fit(dataGenAug.flow(x_train, y_train, batch_size= batch_size_val), validation_data=(x_valid, y_valid), shuffle=1)


# plot 
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('test score', score[0])
print('test accuracy', score[1])


# store model as pickle
pickle_out = open('model trained.p','wb')
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)