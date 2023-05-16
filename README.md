# Self Driving Car - Udacity

## Setup

```
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Usage

In order to use the model and drive the car, run the following command:

```
python .\run.py drive 
```

Optionally, if you want to record the images, run the following command:

```
python .\run.py drive --record folder_name
```

The images will be found in 'recordings\folder_name'.

## Training

As per the instructions, the model was trained on data collected from the first track.  

- The optimizer used is SGD with momentum of 0.9.
- The ReduceLROnPlateau scheduler was used 

The model is fully convolutional, i.e. it does not use linear layers. It uses strided 
convolutions to reduce spatial dimensions. 

It uses Batch norm to make the training stable. The activation function used is ELU.

The final layer uses tanh as the activation function as the output is scaled to `[-1, 1]`. 

The training was performed on Google Colab.
You can find a video of the model running is in this repo.
This project uses the version 2 of the simulator as you suggested on Moodle. 
 

