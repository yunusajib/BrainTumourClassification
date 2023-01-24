import os
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import random
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

path_dir = "C:\\Users\\Laptop\\PycharmProjects\\BrainTumourClassification\\Training"
Categories = ["glioma","meningioma","notumor","pituitary"]

data = []
def create_data():
    for categories in Categories:
        path = os.path.join(path_dir , categories)
        class_name = categories
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img))
                new_array = cv.resize(img_array , (100,100))
                data.append([new_array , class_name])
            except Exception as e:
                pass
create_data()
random.shuffle(data)

x_data = []
y_data = []
for features,labels in data:
    x_data.append(features)
    y_data.append(labels)

df = pd.DataFrame(y_data,columns=["labels"])
x_data = np.array(x_data)

le = LabelEncoder()
y_data = le.fit_transform(y_data)


# load test data
test_data = []
def create_data():
    for categories in Categories:
        path = os.path.join("C:\\Users\\Laptop\\PycharmProjects\\BrainTumourClassification\\Testing", categories)
        class_name = categories
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img))
                new_array = cv.resize(img_array , (100,100))
                test_data.append([new_array , class_name])
            except Exception as e:
                pass
create_data()
random.shuffle(test_data)

x_test = []
y_test = []
for features,labels in test_data:
    x_test.append(features)
    y_test.append(labels)

x_test = np.array(x_test)
le = LabelEncoder()
y_test = le.fit_transform(y_test)

# train model
model = Sequential()
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                 weights=None,
                                                 input_shape=(100,100,3),
                                                 classes=4)
model.add(resnet)
model.add(Flatten())
model.add(Dense(122, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = model.fit(x_data, y_data,validation_data=(x_test,y_test),epochs=20,batch_size=100)
#
# plot training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train","val"])

x_data = x_data.reshape(5712,100,100,3)
x_test = x_test.reshape(1311,100,100,3)

cnn = Sequential()
cnn.add(Conv2D(64,(3,3),activation="relu",input_shape=(100,100,3)))
cnn.add(MaxPool2D(2,2))
cnn.add(Conv2D(128,(5,5),activation="relu"))
cnn.add(MaxPool2D(2,2))
cnn.add(Flatten())
cnn.add(Dense(212,activation="relu"))
cnn.add(Dense(4, activation="softmax"))
cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
history_cnn = cnn.fit(x_data,y_data,validation_data=(x_test,y_test),epochs=20)

plt.plot(history_cnn.history["loss"])
plt.plot(history_cnn.history["val_loss"])
plt.legend(["train","val"])

# prediction
y_pred = cnn.predict(x_test)
#
# # confusion matrix
sns.heatmap(confusion_matrix(y_pred.argmax(axis=1),y_test),annot=True)
confusion_matrix(y_pred.argmax(axis=1),y_test)
accuracy_score(y_pred.argmax(axis=1),y_test)