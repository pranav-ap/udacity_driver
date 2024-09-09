import base64
from io import BytesIO
import os
import shutil
import eventlet
import eventlet.wsgi
import socketio
from PIL import Image
from flask import Flask

# model related imports

import torchvision.transforms as T

from train.model import BabyHamiltonModel
from train.lightning_model import LightningBabyHamiltonModel
from utils.utils import preprocess

# initialize our server
sio = socketio.Server()
app = Flask(__name__)

RECORDINGS_ROOT = '.\\recordings\\'
NEW_RECORDINGS_PATH = ''


# Init empty model
BEST_MODEL_PATH = r'D:\udacity_driver\models\udacity_driver_model.ckpt'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device: {}'.format(device))

model = BabyHamiltonModel()
lightning_model: LightningBabyHamiltonModel = LightningBabyHamiltonModel.load_from_checkpoint(
    BEST_MODEL_PATH,
    model=model
)

lightning_model.eval()

test_transform = T.Compose([
    # T.ToTensor(),  # [0, 255] -> [0, 1]
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # [-0.5, 0.5] -> [-1, 1]
])

# set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 6

# and a speed limit
speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Current Params
        # steering_angle = float(data["steering_angle"])
        # throttle = float(data["throttle"])
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image_orig = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = T.ToTensor()(image_orig)
        image = preprocess(image)
        image = image.float()
        image = test_transform(image)
        image = image.view(-1, 3, 128, 128)
        # image = image.to(device)

        steering_angle_hat = lightning_model(image)
        steering_angle_hat = steering_angle_hat.item()
        # click.echo('Steering angle : {}'.format(steering_angle_hat))

        global speed_limit
        speed_limit = MIN_SPEED if speed > speed_limit else MAX_SPEED

        throttle = 1.0 - steering_angle_hat ** 2 - (speed / speed_limit) ** 2
        # click.echo('Throttle : {}'.format(throttle))

        send_control(steering_angle_hat, throttle)

        # save frame
        # if NEW_RECORDINGS_PATH != '':
        #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #     image_filepath = os.path.join(NEW_RECORDINGS_PATH, timestamp)
        #     image_orig.save('{}.jpg'.format(image_filepath))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("Connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


def start(record):
    if record != '':
        global NEW_RECORDINGS_PATH
        NEW_RECORDINGS_PATH = RECORDINGS_ROOT + record

        if os.path.exists(NEW_RECORDINGS_PATH):
            shutil.rmtree(NEW_RECORDINGS_PATH)

        os.makedirs(NEW_RECORDINGS_PATH)

    global app
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
