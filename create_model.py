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
import time

start = time.time()
batchSize = 5
trainingEpochs = 40

model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

trainingHistory = model.fit(total_input, total_output, batch_size=batchSize,
                            epochs=trainingEpochs,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            shuffle=True,)

trainingAccuracy = trainingHistory.history['acc']
validationAccuracy = trainingHistory.history['val_acc']

end = time.time()
print("*Net Time : ", (end-start)/60, ' mins...')

print("Training Accuracy "+str(trainingAccuracy[-1]))
print("Validation Accuracy "+str(validationAccuracy[-1]))

model.save_weights('model_weights.h5')
