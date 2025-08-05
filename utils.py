from torchvision import transforms as T
from datetime import datetime
from enum import Enum
from tokyy import CHECKPOINTS_DIR
from pathlib import Path
import argparse
import os


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
            print( "Please enter Y or N." )


def log_message( type, msj = '' ):
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

    return type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "--no_ask_before", action = "store_true", help = "Disable confirmation prompts in Trainer")

    parser.add_argument( "--checkpoint_dir", type = Path, default = CHECKPOINTS_DIR, help = "Path to the checkpoint directory. Default at /checkpoints" )
    parser.add_argument( "--checkpoint_name", type = str, default = "default.pt", help = "Checkpoint file name." )

    parser.add_argument( "--image_size", nargs = 2, type = int, default = [ 128, 128 ], help = "Image size (height width).")
    parser.add_argument( "--batch_size", type = int, default = 32, help = "Training batch size." )
    parser.add_argument( "--accum_steps", type = int, default = 2, help = "Gradient accumulation steps." )

    parser.add_argument( "--architecture", type = str, default = "resunet", help = "Architecture of the model to train [resunet, cbam, atrous]." )

    args = parser.parse_args()

    args.checkpoint_path = args.checkpoint_dir / args.checkpoint_name
    log_message( LogType.NONE, f"Checkpoint absolute path is { args.checkpoint_path }" )
    args.image_size = tuple( args.image_size )
    log_message( LogType.NONE, f"Model will begin training on input size (re)shaped to { args.image_size }" )
    log_message( LogType.NONE, f"Dataset loaders batch size set to { args.batch_size }" )
    log_message( LogType.NONE, f"Gradient number of accumulation steps set to { args.accum_steps }" )

    if  not args.checkpoint_name.endswith( ( ".pt", ".pth", ".ckpt" ) ):
        log_message( LogType.WARNING, "It is recommended that the provided checkpoint file name ends with .pt, .pth, or .ckpt" )
        ask_yes_no( "Continue?" )

    return args

def parse_test_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument( "--show", action = "store_true", help = "Traverse dataset." )
    parser.add_argument( "--batch", action = "store_true", help = "Calculate max batch size on without raising OOM." )
    parser.add_argument( "--dataset", action = "store_true", help = "Print dataset data." )
    parser.add_argument( "--predict", action = "store_true", help = "Save 8 predictions and their ground truths." )
    parser.add_argument( "--graphs", action = "store_true", help = "Print dataset data." )
    parser.add_argument( "--one", action = "store_true", help = "Predict test.png" )

    parser.add_argument( "--checkpoint_dir", type = Path, default = CHECKPOINTS_DIR, help = "Path to the checkpoint directory. Default at /checkpoints" )
    parser.add_argument( "--checkpoint_name", type = str, default = "default", help = "Checkpoint file name." )

    parser.add_argument( "--architecture", type = str, default = "resunet", help = "Architecture of the model to train [resunet, cbam, atrous]." )

    args = parser.parse_args()

    args.checkpoint_path = args.checkpoint_dir / f"{ args.checkpoint_name }.pt"
    log_message( LogType.NONE, f"Checkpoint absolute path is { args.checkpoint_path }" )

    if args.checkpoint_name.endswith( ( ".pt", ".pth", ".ckpt" ) ):
        log_message( LogType.ERROR, "The provided checkpoint should not end with .pt, .pth, or .ckpt" )
        return None 
    
    return args
