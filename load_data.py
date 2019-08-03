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


def load_data():
    input_images=[]             # List of all input images
    target_images=[]            # List of all output images
    for i,image in enumerate(os.listdir('flower_images')):               # Iterating through downloaded dataset
        if image[-3:]=='png':
            img=cv2.imread('flower_images/'+image)
            img=np.resize(img,(128,128,3))                               # Resizing all images to required Size
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                      # OpenCv reads image in BGR format. Here we convert to RGB

            img1=img/255.0                                               # Normalizing mage (Important)
        #print(img1.shape)
            input_images.append(img1)                                    # Adding Image to list
            img2=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)                    # Converting RGB to GRAY
            img2=img2/255.0
            target_images.append(img2)                                   # Adding img to output list
        #print(img2.shape)

    total_output=np.array(target_images)[:,:,:,np.newaxis]               # Shape of GRAY img is (128,128). We add new axis
                                                                     # New shape (128,128,1)
    total_input=np.array(input_images)                                   # Converting it to array

    return total_output,total_input
