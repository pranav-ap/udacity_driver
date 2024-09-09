from logger_setup import logger
from moviepy.editor import ImageSequenceClip
import os


def start(record, fps):
    RECORDINGS_ROOT = '.\\recordings\\'
    IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']

    NEW_RECORDINGS_PATH = RECORDINGS_ROOT + record

    image_list = sorted([
        os.path.join(NEW_RECORDINGS_PATH, image_file)
        for image_file in os.listdir(NEW_RECORDINGS_PATH)
    ])

    image_list = [
        image_file
        for image_file in image_list
        if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT
    ]

    logger.info(f"Creating video {record}, FPS={fps}")
    clip = ImageSequenceClip(image_list, fps=fps)

    video_file = record + '.mp4'
    clip.write_videofile(NEW_RECORDINGS_PATH + video_file)

