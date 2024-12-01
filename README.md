# Driving Behavior Cloning 

## Overview

The the goal of behavioral cloning is to collect data while 
exhibiting good behavior and then train a model to mimic that behavior.

This project uses a simulator to collect data to 
train a model to drive the car autonomously. The simulator was released by Udacity for its
Self-Driving Car Nanodegree. 

## Setup

```
pip install -r requirements.txt

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Usage

To use the model and drive the car, run the following command:

```
python .\run.py drive 
```

Optionally, if you want to record the images, run the following command:

```
python .\run.py drive --record folder_name
```

The images will be found in 'recordings\folder_name'.

## Training

The simulator supports two tracks. The model was trained on data 
collected from the first track. The second track is used for testing. 

Since most of the turns in the tracks are left turns, we have
an imbalanced dataset. To address this, I used data augmentation
to generate more right turns by using image flips, along with other standard
augmentations like adding noises, brightness, and contrast adjustments.

The model used is based on the one used in the NVIDIA End to End Learning 
for Self-Driving Cars paper. 

The model is fully convolutional and uses Batch norm to make the training stable. 
The activation function used is ELU. The final layer is a tanh activation function 
because it scales the output to `[-1, 1]`.

The optimizer used is SGD with momentum of 0.9 along with the 
ReduceLROnPlateau scheduler. 

