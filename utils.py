from torchvision import transforms as T
from datetime import datetime
from enum import Enum
from tokyy import CHECKPOINTS_DIR
from pathlib import Path
import argparse
import sys


class LogType( Enum ):
    SUCCESS = 1
    ERROR = -1
    NONE = 0
    WARNING = 2
    OK = None

def ask_yes_no( question ):
    while True:
        answer = input( question + " [Y/n]: " ).strip().lower()
        if answer in ( 'y', 'yes', 'Y' ):
            return True
        elif answer in ( 'n', 'no', 'N' ):
            return False
        else:
            print( "Invalid command." )


def log_message( type, msj = '', exit = False ):
    if type == LogType.ERROR:
        print( '\033[31m[  ERROR  ]\033[0m', end = ' ' )

    elif type == LogType.SUCCESS:
        print( '\033[32m[ SUCCESS ]\033[0m', end = ' ' )

    elif type == LogType.OK:
        print( '\033[32m[ OK ]\033[0m     ', end = ' ' )
    
    elif type == LogType.WARNING:
        print( '\033[33m[ WARNING ]\033[0m', end = ' ')

    print( '\033[34m', msj, '\033[0m', end = ' - ' )

    now = datetime.now()

    print( now.strftime( "%H:%M:%S" ) )

    if exit == True:
        sys.exit( 1 )

    return type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "--no-ask-before", action = "store_true", help = "Disable confirmation prompts in Trainer")
    parser.add_argument( "--no-scheduler", action = "store_true", help = "Don't use scheduler.")

    parser.add_argument( "--checkpoint-dir", type = Path, default = CHECKPOINTS_DIR, help = "Path to the checkpoint directory. Default at /checkpoints" )
    parser.add_argument( "--checkpoint-name", type = str, default = "default.pt", help = "Checkpoint file name." )

    parser.add_argument( "--image-size", nargs = 2, type = int, default = [ 128, 128 ], help = "Image size (height width).")
    parser.add_argument( "--batch-size", type = int, default = 32, help = "Training batch size." )
    parser.add_argument( "--accum-steps", type = int, default = 1, help = "Gradient accumulation steps." )
    parser.add_argument( "--max-epochs", type = int, default = 5, help = "Training maximum epochs." )

    parser.add_argument( "--architecture", type = str, default = "cbam", help = "Architecture of the model to train [resunet, cbam, atrous]." )

    args = parser.parse_args()

    args.checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    log_message( LogType.NONE, f"Checkpoint absolute path is { args.checkpoint_path }" )
    name_to_architecture = {
        "unet" : "ResUNet",
        "cbam" : "ResCBAMUNet",
        "atrous" : "AtrousResCBAMUNet"
    }
    log_message( LogType.NONE, f"Model architecture is set to { name_to_architecture[ args.architecture ] }" )
    args.image_size = tuple( args.image_size )
    log_message( LogType.NONE, f"Model will begin training on input size (re)shaped to { args.image_size }" )
    log_message( LogType.NONE, f"Dataset loaders batch size set to { args.batch_size }" )
    log_message( LogType.NONE, f"Gradient number of accumulation steps set to { args.accum_steps }" )

    if  not args.checkpoint_name.endswith( ( ".pt", ".pth", ".ckpt" ) ):
        log_message( LogType.WARNING, "It is recommended that the provided checkpoint file name ends with .pt, .pth, or .ckpt" )
        ask_yes_no( "Continue?" )

    return args

def parse_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "--all", action = "store_true", help = "Plot all." )
    parser.add_argument( "--lr", action = "store_true", help = "Plot learning rate." )
    parser.add_argument( "--pred", action = "store_true", help = "Plot predictions." )
    parser.add_argument( "--loss", action = "store_true", help = "Plot losses." )
    parser.add_argument( "--_loss", action = "store_true", help = "Plot _losses." )
    parser.add_argument( "--metric", action = "store_true", help = "Plot metrics." )

    parser.add_argument( "--delete-by-number", type = str, help = "Delete plots by number." )
    parser.add_argument( "--delete-by-suffix", type = str, help = "Delete plots by suffix." )

    parser.add_argument( "--checkpoint-dir", type = Path, default = CHECKPOINTS_DIR, help = "Path to the checkpoint directory. Default at /checkpoints" )
    parser.add_argument( "--checkpoint-name", type = str, default = "sicucoaiele.pt", help = "Checkpoint file name." )

    parser.add_argument( "--arch", type = str, default = "cbam", help = "Architecture of the model to train [resunet, cbam, atrous]." )

    args = parser.parse_args()

    args.checkpoint_path = args.checkpoint_dir / f"{ args.checkpoint_name }"
    log_message( LogType.NONE, f"Checkpoint absolute path is { args.checkpoint_path }" )

    return args

def get_new_file_number( dir_path : Path ) -> str:
    max_number = 0
    has_file = False

    for file in dir_path.iterdir():
        if file.is_file():
            has_file = True  
            number = int( file.name[ : 2 ] )
            if max_number < number: max_number = number

    if has_file is False:
        return "00"

    max_number += 1

    if max_number < 10: 
        return ( "0" + str( max_number ) )
    
    return str( max_number )
