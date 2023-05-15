import click
import drive
import video


@click.group()
def cli():
    pass


@cli.command('drive')
@click.option('--record', default='', help='Where to store the recorded images?')
def drive_cmd(record: str):
    click.echo('Begin Driving!')
    drive.start(record)


@cli.command('video')
@click.option('--record', default='', required=True, help='Path to image folder. The video will be created from these images.')
@click.option('--fps', default=60, help='FPS (Frames per second) setting for the video.')
def video_cmd(record: str, fps: int):
    click.echo('Begin Making Video!')
    video.start(record, fps)


if __name__ == '__main__':
    cli()
