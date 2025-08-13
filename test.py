from tokyy.utils import LogType, log_message, get_new_file_number
# from tokyy.models.resunet2 import ResUNet
from tokyy.models.vision import ResUNet, ResCBAMUNet, AtrousResCBAMUNet
import tokyy.datasets.vision
from tokyy.datasets.vision import NyuDepthV2 
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metrics
from tokyy import  LOSSES_DIR, METRICS_DIR, PREDICTS_DIR, OTHERS_DIR, LEARNING_RATES_DIR, _LOSSES_TRAIN_DIR, _LOSSES_TEST_DIR, _LOSSES_VAL_DIR
from tokyy.augmentation import Resize, RandomCrop, RandomHorizontalFlip, ToTensor, ColorJitter, PairedCompose, RandomRotationByAngle
from tokyy.utils import parse_plot_args
from tokyy.losses.vision import DepthLossV2

from torchvision import transforms as T
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import matplotlib
import matplotlib.pyplot as plt
 
import numpy as np

from typing import Dict, Tuple

import os

matplotlib.use( 'TkAgg' )

image_size = ( 128, 128 )
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

from tokyy.plotter import Plotter


def load_model_and_checkpointer( model, checkpoint_path ) -> Tuple[ torch.nn.Module, Checkpointer ]:
    cp = Checkpointer.load(
        model = model,
        optimizer = torch.optim.Adam( model.parameters(), lr = 1e-4 ),
        criterion = DepthLossV2(),
        scaler = GradScaler(),
        scheduler =  torch.optim.lr_scheduler.OneCycleLR( 
            torch.optim.Adam( model.parameters(), lr = 1e-4 ), 
            max_lr = 1e-3,             
            steps_per_epoch = 743,
            epochs = 5,                   
            anneal_strategy = 'cos',       
            pct_start = 0.3,
            div_factor = 25.0,             
            final_div_factor = 1e4,
        ),
        
        map_loc = 'cpu',
        path = checkpoint_path
    )

    if cp is None:
        return ( None, None )

    model = cp.model.to( torch.device( 'cpu' ) ) 
    model.eval()

    return ( model, cp )


def batch( model : torch.nn.Module ):
    batch_size = 4
    while True:
        try:
            dummy = torch.randn( batch_size, 3, 128, 128 ).cuda()
            model( dummy )
            log_message( LogType.SUCCESS, f"Batch size of { batch_size } did not raise OOM" )
            batch_size += 4

        except RuntimeError:
            log_message( LogType.ERROR, f"OOM raised at batch size of { batch_size }" )
            break


def dataset():
    batch_size = 32
    # n_samples = 2_000

    train_dataset = NyuDepthV2( "/home/TokYy/DL_Datasets/nyu/train", transform = tokyy.datasets.vision.rgb_transform, depth_transform = tokyy.datasets.vision.depth_transform )
    test_dataset = NyuDepthV2( "/home/TokYy/DL_Datasets/nyu/test", transform = tokyy.datasets.vision.rgb_transform, depth_transform = tokyy.datasets.vision.depth_transform )

    train_loader = DataLoader( train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True )
    test_loader = DataLoader( test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True )

    log_message( LogType.NONE, f"The train dataset has { train_dataset.n_scenes } scenes" )
    log_message( LogType.NONE, f"The test dataset has { test_dataset.n_scenes } scenes" )

    log_message( LogType.NONE, f"Loaded { len( train_loader.dataset ) } training samples" )
    log_message( LogType.NONE, f"Loaded { len( test_loader.dataset ) } training samples" )

    log_message( LogType.NONE, f"With a batch size of { batch_size } ==> { len( train_loader ) } batches for training" )
    log_message( LogType.NONE, f"With a batch size of { batch_size } ==> { len( test_loader ) } batches for testing" )

    # min_val = float( 'inf' )
    # max_val = float( '-inf' )

    # for i in tqdm( range( n_samples ) ):
    #     _, depth = train_loader.dataset[ i ]
        
    #     depth = depth.float()

    #     batch_min = depth.min().item()
    #     batch_max = depth.max().item()

    #     min_val = min( min_val, batch_min )
    #     max_val = max( max_val, batch_max )

    # log_message( LogType.WARNING, f"Depth Min and Max extracted from the first { n_samples } samples" )
    # log_message( LogType.NONE, f"Depth Min: { min_val }" )
    # log_message( LogType.NONE, f"Depth Max: { max_val }" )


def show_one_sample( rgb_tensor, depth_tensor ):
    if rgb_tensor.shape[ 0 ] == 3:
        rgb_tensor = np.transpose( rgb_tensor, ( 1, 2, 0 ) )

    rgb_img = rgb_tensor.cpu().numpy()
    depth_tensor = depth_tensor.squeeze().cpu().numpy()

    log_message( LogType.NONE, f"RGB min: { rgb_tensor.min() }, max: { rgb_tensor.max() }" )
    log_message( LogType.NONE, f"Depth min: { depth_tensor.min() }, max: { depth_tensor.max() }" )

    log_message( LogType.NONE, f"RGB shape: { rgb_tensor.shape }" )
    log_message( LogType.NONE, f"Depth shape: { depth_tensor.shape }" )

    plt.clf()
    plt.subplot( 1, 2, 1)
    plt.title( "RGB Image" )
    plt.imshow( rgb_img )  
    plt.axis( 'off' )

    plt.subplot( 1, 2, 2 )
    plt.title( "Depth Map" )
    plt.imshow( depth_tensor, cmap = 'plasma' )
    plt.colorbar( label = 'Depth' )
    plt.axis( 'off' )


def show():
    paired_transform = PairedCompose([
        Resize( ( 144, 192 ) ),
        RandomRotationByAngle(),
        RandomHorizontalFlip(),
        RandomCrop( ( 128, 128 ) ),
        ColorJitter(),
        ToTensor()
    ])

    dataset = NyuDepthV2( "/home/TokYy/DL_Datasets/nyu/test", paired_transform )

    for rgb, depth in dataset:
        show_one_sample( rgb, depth )
        log_message( LogType.NONE, "Press any key to see next image, or close window ... CTRL ^ C in terminal to exit" )
        if plt.waitforbuttonpress():
            continue


def main():
    name_to_architecture = {
        "resunet" : ResUNet,
        "cbam" : ResCBAMUNet,
        "atrous" : AtrousResCBAMUNet
    }

    args = parse_plot_args()

    suffix = os.path.splitext( args.checkpoint_name )[ 0 ]

    if args.delete_by_number is not None:
        Plotter.delete_by_nubmer( args.delete_by_number )

    if args.delete_by_suffix is not None:
        Plotter.delete_by_suffix( args.delete_by_suffix )

    model, checkpointer = load_model_and_checkpointer( name_to_architecture[ args.arch ](), args.checkpoint_path )

    plotter = Plotter( model, checkpointer )

    plot_actions : Dict[ callable, bool] = {
        plotter.predict : args.pred or args.all,
        plotter.plot__losses : args._loss or args.all,
        plotter.plot_learning_rates : args.lr or args.all,
        plotter.plot_metrics : args.metric or args.all,
        plotter.plot_losses : args.loss or args.all,
    }

    for act, flag_set in plot_actions.items():
        if flag_set:
            act( suffix )


if __name__ == '__main__':
    main()
