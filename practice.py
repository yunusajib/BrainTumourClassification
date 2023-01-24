import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

data = []
labels =[]
categories = ['cats', 'dogs']
Dir= 'C:\\Users\\Laptop\\Desktop\\training_set\\training_set'

for category in categories:
    path = os.path.join(Dir, category)
    label = categories
    for img in os.listdir(path):
         img = cv2.imread(os.path.join(path, img))
         img = cv2.resize(img, (224, 224))
         data.append(img)
         labels.append(label)

data = np.array(data)
label = np.array(labels)

print(data.shape)



