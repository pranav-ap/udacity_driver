import click


@click.group()
def cli():
    pass


@cli.command('drive')
@click.option('--record', default='', help='Name of new recording')
def drive_cmd(record: str):
    click.echo('Begin Driving!')
    import drive
    drive.start(record)


@cli.command('video')
@click.option('--record', default='', required=True, help='Name of recording')
@click.option('--fps', default=60, help='FPS (Frames per second) setting for the video.')
def video_cmd(record: str, fps: int):
    click.echo('Begin Making Video!')
    import video
    video.start(record, fps)


if __name__ == '__main__':
    cli()
