# Autoencoder_RGB2GRAY
# Please Refer .ipynb file for more detailed version of this project
# RGB to Grayscale Using Auto-encoder

**I have used flower color images from kaggle dataset. You can download and extract it to current directory from [here](https://www.kaggle.com/olgabelitskaya/flower-color-images)**

> flower_images/*.png

# Dataset Information
**It consists of 210 total image data. (I could have used bigger dataset but unfortunately my internet is too slow today.)**<br>
**The image shape is (128,128,3)**

# Objective
**To convert RGB images to Grayscale using Neural Network**<br>

# Autoencoders
**An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”.<br>
Here we have used CNN to learn features and target the grayscale images**

### You need to install following libraries
- Tensorflow
- Keras
- OpenCV
- Numpy

You can install them by:
```
pip install <package-name>
```

You can refer to the .ipynb file for more detailed on how this project is done.<br>
**Remove the commented lines to include the libraries**<br>
-**run_mode.py---To run directly the model**<br>
-**load_data.py---Loads the data**<br>
-**create_model.py----Created the model**<br>


