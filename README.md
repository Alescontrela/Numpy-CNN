# Numpy-CNN
A numpy-only implementation of a Convolutional Neural Network, from the ground up.

*Written by Alejandro Escontrela for [this article](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1) on CNNs.*

## Purpose

To gain a quality understanding of convolutional neural networks and what makes them peform so well, I constructed one from scratch with NumPy. This CNN is in no way intended to replace popular DL frameworks such as Tensorflow or Torch, *it is instead meant to serve as an instructional tool.*

## Training the network

To train the network on your machine, first install all necessary dependencies using:


`$ pip install -r requirements.txt`

Afterwards, you can train the network using the following command: 

`$ python3 train_cnn.py '<file_name>.pkl'`

Replace `<file_name>` with whatever file name you would like. The terminal should display the following progress bar to indicate the training progress, as well as the cost for the current training batch:
<p align = "center">
<img width="80%" alt="portfolio_view" src="https://github.com/Alescontrela/Numpy-CNN/blob/master/images/training_progress.png">
</p>


After the CNN has finished training, a .pkl file containing the network's parameters is saved to the directory where the script was run.

The network takes about 5 hours to train on my macbook pro. I included the trained params in the GitHub repo under the name `params.pkl` . To use the pretrained params when measuring the network's performance, replace `<file_name>` with params.pkl.

## Measuring Performance
To measure the network's accuracy, run the following command in the terminal:

`$ python3 measure_performance.py '<file_name>.pkl'`

This command will use the trained parameters to run predictions on all 10,000 digits in the test dataset. After all predictions are made, a value displaying the network's accuracy will appear in the command prompt:

<p align="center">
<img width="80%" alt="portfolio_view" src="https://github.com/Alescontrela/Numpy-CNN/blob/master/images/test_accuracy.png">
</p>
