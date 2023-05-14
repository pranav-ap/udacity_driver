# networking and other boring stuff

import base64
import os
import shutil
from datetime import datetime
from io import BytesIO

# utility functions
import click
import eventlet
import eventlet.wsgi
import socketio
import torch
from PIL import Image
from flask import Flask

from model import BabyHamiltonModel

# main imports

# initialize our server
sio = socketio.Server()
app = Flask(__name__)

# Init empty model 
model = None


# Paths
RECORD_PATH = './record/'
MODELS_PATH = './models/'
new_record_folder_path = ''


#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Current Params
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        try:
            # predict the steering angle for the image
            steering_angle = model(img)
            
            steering_angle = steering_angle.item()
            click.echo('Steering angle : {}'.format(steering_angle))
            
            throttle = 1.2 - steering_angle ** 2 - (speed / set_speed) ** 2
            click.echo('Throttle : {}'.format(throttle))      
            
            click.echo('Speed : {}'.format(speed))
            
            send_control(steering_angle, throttle)
            
        except Exception as e:
            click.echo(e)

        # save frame
        if new_record_folder_path != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_path = os.path.join(new_record_folder_path, timestamp)
            image.save('{}.jpg'.format(image_path))
    else:
        sio.emit('manual', data={}, skip_sid=True)


# Function to connect and communiucate with simulator socket.
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


# Send the generated steering angle and acceleration value to the simulator
def send_control(steering_angle, throttle):
    sio.emit("steer", data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        }, skip_sid=True)
    

def start(model_name, record, clear):
    # load model
    global model
    model = BabyHamiltonModel()

    if torch.cuda.is_available():
        model = model.cuda()

    model_path = os.path.join(MODELS_PATH, model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if clear:
        shutil.rmtree(RECORD_PATH)
        click.echo("Cleared record folder ...")

    if record:
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        new_record_folder_path = os.path.join(RECORD_PATH, timestamp)
        os.makedirs(new_record_folder_path)
        click.echo("Recording this run at {} ...".format(new_record_folder_path))

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

