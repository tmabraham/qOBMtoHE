import argparse
import os
from fastgpu import *


def get_training_commands(filename):
    with open(filename) as f:
        lines = f.readlines()
    commands = [command for command in lines if command.startswith('python')]
    return commands

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands_file', type=str, required=True, help='File with commands to run')
    args = parser.parse_args()
    
    # Create scripts from list of commands
    commands = get_training_commands(args.commands_file)
    try: 
        os.makedirs('to_run')
    except: 
        print('previous experiments not done, exiting...'); exit()
    print('creating scripts...')
    for i, command in enumerate(commands):
        with open(f'to_run/train_{str(i).zfill(4)}.sh', 'w') as f:
            f.write('#!/bin/sh\n')
            f.write(command)

    os.system('chmod +x to_run/*.sh')

    # fastgpu
    print('running scripts...')
    rp = ResourcePoolGPU(path='.')
    rp.poll_scripts(exit_when_empty=1)


