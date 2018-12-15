# Introduction
This project is built to use deep learning methods to identify 102 different flower species found in UK.The dataset for this project is taken from this website.
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

This dataset includes 102 different categories of images which is being used to train the CNN models.The dataset is being split into train,test and validation sets.The ouput of the application would be the probability of the predicted species .

# Dependencies and Modules
This program used pytorch's libraries for working with these images.A GPU would be necessary to train the model on CUDA and run as the time taken would be significantly lower if done on a GPU.Although the program has an option to do training on normal CPU mode use of a GPU is highly recommended.

# This application has been divided into 4 different files :

- define_network.py
- predict.py
- train.py
- utils.py


# Model Options available to train
- VGG16
- VGG13
- DENSENET121

The default algorithm being used is densenet121 with which 89% accuracy is being achieved on the flowers dataset
