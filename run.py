# import subprocess

# if __name__ == '__main__':
#     # subprocess.call(['python', 'train.py', '-d', '/home/yang/git/udacity_test/End-to-End-Learning-for-Self-Driving-Cars-master/data'])
#     subprocess.call(['python', 'drive.py', 'models/model_390000'])

import click
import drive
import train

@click.group()
def cli():
    pass

@cli.command('drive')
@click.option('--model', default='', help='Name of Model')
@click.option('--record', is_flag=True, help='Record and store frames of simulation?')
@click.option('--clear', is_flag=True, help='Clear Previous recordings?')
def drive_cmd(model:str, record:bool, clear:bool):
    click.echo('Begin Driving!')
    drive.start(model, record, clear)

@cli.command('train')
@click.option('--model', default='', help='Name of Model')
def train_cmd(model:str):
    click.echo('Begin Training!')
    train.start(model)

if __name__ == '__main__':
    cli()

