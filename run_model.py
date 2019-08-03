# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# #%matplotlib inline
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# from keras.utils import plot_model,to_categorical
# import os
# import cv2
# from keras.layers import UpSampling2D,Input
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 10

from load_data import load_data
total_output,total_input=load_data()

total_output,total_input=load_data()

def autoencoder():
    model=Sequential()

    #Encoder
    model.add(Conv2D(64,3,padding='same',activation='relu',input_shape=(128,128,3)))
    model.add(MaxPooling2D((2,2),padding='same'))
    #Decoder
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(1,2,padding='same',activation='relu'))


    print(model.summary())
    return model
model=autoencoder()
model.load_weights('model_weights.h5')

idx=15
image=total_input[idx]                              # Sample Image
output=total_output[idx]
img2=model.predict(image.reshape(1,128,128,3))
print(img2.shape)
cv2.imshow('Predicted',np.reshape(img2,(128,128,1)))
cv2.imshow('Original',np.reshape(image,(128,128,3)))
cv2.imwrite('output.png',np.reshape(img2,(128,128,1)))
cv2.waitKey(1000)
cv2.destroyAllWindows()
