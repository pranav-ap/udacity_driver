from moviepy.editor import ImageSequenceClip
import os


RECORDINGS_ROOT = './recordings/'
NEW_RECORDINGS_PATH = ''
IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def start(record, fps):
    global NEW_RECORDINGS_PATH
    NEW_RECORDINGS_PATH = RECORDINGS_ROOT + record

    image_list = sorted([os.path.join(record, image_file)
                         for image_file in os.listdir(record)])

    image_list = [image_file for image_file in image_list if os.path.splitext(
        image_file)[1][1:].lower() in IMAGE_EXT]

    print("Creating video {0}, FPS={1}".format(record, fps))
    clip = ImageSequenceClip(image_list, fps=fps)

    video_file = record + '.mp4'
    clip.write_videofile(video_file)

