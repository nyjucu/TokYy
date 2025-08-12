from tokyy.utils import LogType, log_message, get_new_file_number
# from tokyy.models.resunet2 import ResUNet
from tokyy.models.vision import ResUNet, ResCBAMUNet, AtrousResCBAMUNet
import tokyy.datasets.vision
from tokyy.datasets.vision import NyuDepthV2 
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metrics
from tokyy import  LOSSES_DIR, METRICS_DIR, PREDICTS_DIR, OTHERS_DIR, LEARNING_RATES_DIR, _LOSSES_TRAIN_DIR, _LOSSES_TEST_DIR, _LOSSES_VAL_DIR
from tokyy.augmentation import Resize, RandomCrop, RandomHorizontalFlip, ToTensor, ColorJitter, PairedCompose
from tokyy.utils import parse_test_args
from tokyy.losses.vision import DepthLossV2

from torchvision import transforms as T
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import matplotlib
import matplotlib.pyplot as plt

import h5py
import numpy as np

from PIL import Image
from typing import Dict

import os

matplotlib.use( 'TkAgg' )

image_size = ( 128, 128 )
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )


def load_model_and_checkpointer( model, checkpoint_path ) -> tuple:
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

    model = cp.model.to( device ) 
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


def predict( suffix : str, model : torch.nn.Module ):
    predicts_path = PREDICTS_DIR / ( get_new_file_number( PREDICTS_DIR ) + "_predicts_" + suffix )

    file_paths = [
        r"/home/TokYy/DL_Datasets/nyu/test/01149.h5",
        r"/home/TokYy/DL_Datasets/nyu/test/00002.h5",
        r"/home/TokYy/DL_Datasets/nyu/test/00009.h5",
        r"/home/TokYy/DL_Datasets/nyu/test/00169.h5",

        r"/home/TokYy/DL_Datasets/nyu/train/bedroom_0051/00001.h5",
        r"/home/TokYy/DL_Datasets/nyu/train/classroom_0003/00001.h5",
        r"/home/TokYy/DL_Datasets/nyu/train/dining_room_0033/00001.h5",
        r"/home/TokYy/DL_Datasets/nyu/train/office_0025/00001.h5",
    ]

    rgb_transform = T.Compose( [
        T.ToTensor(),
        T.Resize( image_size ),
    ] )

    depth_transform = T.Compose( [
        T.ToTensor(),
        T.Resize( image_size ),
    ] )

    n = len( file_paths )
    _, axs = plt.subplots( n, 3, figsize = ( 12, 4 * n ) )
    
    for i, file_path in enumerate( file_paths ):            
        with h5py.File( file_path, 'r' ) as f:
            rgb = np.array( f[ 'rgb' ] )
            depth = np.array( f[ 'depth' ] )

        if rgb.shape[ 0 ] == 3:
            rgb = np.transpose( rgb, ( 1, 2, 0 ) )

        rgb_tensor = rgb_transform( rgb ).unsqueeze( 0 ).to( device )
        depth_tensor = depth_transform( depth ).squeeze().cpu().numpy() / 10.0

        with torch.no_grad():
            pred = model( rgb_tensor )
            pred = pred.squeeze().cpu().numpy() 

        axs[ i ][ 0 ].imshow( rgb.astype( np.uint8 ) )
        axs[ i ][ 0 ].set_title( f"RGB Image [{ i }]" )
        axs[ i ][ 0 ].axis( "off" )

        axs[ i ][ 1 ].imshow( depth_tensor, cmap = 'plasma' )
        axs[ i ][ 1 ].set_title( "Ground Truth Depth" )
        axs[ i ][ 1 ].axis( "off" )

        axs[ i ][ 2 ].imshow(pred, cmap='plasma')
        axs[ i ][ 2 ].set_title("Predicted Depth")
        axs[ i ][ 2 ].axis("off")

    plt.tight_layout()
    plt.savefig( predicts_path )

    log_message( LogType.OK, f"Preictions saved at { predicts_path }" )


def plot_metrics( suffix : str, metrics : dict ):
    metrics_path =  METRICS_DIR / ( get_new_file_number( METRICS_DIR ) + "_metrics_" + suffix)

    plt.figure( figsize = ( 12, 6 ) )

    for metric, values in metrics.items():
        if isinstance(metric, Metrics):
            cpu_values = [
                x.detach().cpu().item() if torch.is_tensor( x ) else float( x )
                for x in values
            ]

            plt.plot( cpu_values, label = metric.name )

    plt.title( "Metrics Over Epochs" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Metric Value" )
    plt.legend()
    plt.grid( True )
    plt.tight_layout()
    plt.savefig( metrics_path )
    plt.close()
    
    log_message( LogType.SUCCESS, f"Metrics graph saved to { metrics_path }" )


def plot_losses( suffix : str, losses : dict ):
    losses_path = LOSSES_DIR / ( get_new_file_number( LOSSES_DIR ) + "_losses_" + suffix )

    plt.figure( figsize = ( 12, 6 ) )

    for loss_type, values in losses.items():
        cpu_values = [
            x.detach().cpu().item() if torch.is_tensor(x) else float(x)
            for x in values
        ]

        plt.plot( cpu_values, label=loss_type )

    plt.title( "Training and Validation Loss Over Epochs" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Loss" )
    plt.legend()
    plt.grid( True )
    plt.tight_layout()
    plt.savefig( losses_path )
    plt.close()

    log_message( LogType.SUCCESS, f"Losses graph saved to { losses_path }" )


def plot__losses( suffix : str, _losses : dict, sets : list = [ 'train', 'val', 'test' ] ):
    set_to_path = {
        "train" : _LOSSES_TRAIN_DIR / ( get_new_file_number( _LOSSES_TRAIN_DIR ) + "__losses_train_" + suffix ),
        "test" :  _LOSSES_TEST_DIR / ( get_new_file_number( _LOSSES_TEST_DIR ) + "__losses_test_" + suffix ),
        "val" : _LOSSES_VAL_DIR / ( get_new_file_number( _LOSSES_VAL_DIR ) + "__losses_val_" + suffix )
    }

    for set in sets:
        plt.figure( figsize = ( 12, 6 ) )

        for _losses_set, _losses_dict in _losses.items():
            if _losses_set == set :
                for loss, values in _losses_dict.items():
                    cpu_values = [
                        x.detach().cpu().item() if torch.is_tensor( x ) else float( x )
                        for x in values
                    ] 

                    plt.plot( cpu_values, label = loss )

        plt.title( f"Loss Over Epochs : { set }" )
        plt.xlabel( "Epoch" )
        plt.ylabel( "Loss Value" )
        plt.legend()
        plt.grid( True )
        plt.tight_layout()
        plt.savefig( set_to_path[ set ] )
        plt.close()
        
        log_message( LogType.SUCCESS, f"_Losses graph saved to { set_to_path[ set ] }" )


def plot_learning_rates( suffix : str, learning_rates : list ):
    learning_rates_path = LEARNING_RATES_DIR / ( get_new_file_number( LEARNING_RATES_DIR ) + "_lr_" + suffix )

    plt.figure( figsize = ( 12, 6 ) )

    cpu_values = [
        x.detach().cpu().item() if torch.is_tensor( x ) else float( x )
        for x in learning_rates
    ]

    plt.plot( cpu_values, label = "learning rate" )

    plt.title( "Learning Rate Over Epochs" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Learning Rate Value" )
    plt.legend()
    plt.grid( True )
    plt.tight_layout()
    plt.savefig( learning_rates_path )
    plt.close()
    
    log_message( LogType.SUCCESS, f"Learning rates graph saved to { learning_rates_path }" )


def graphs( suffix : str, checkpointer ):
    plot_metrics( suffix, checkpointer.metrics )
    plot_losses( suffix, checkpointer.losses )
    plot__losses( suffix, checkpointer._losses )
    plot_learning_rates( suffix, checkpointer.learning_rates )


def one( suffix : str, model : torch.nn.Module, image_path = r'/home/TokYy/PyTorchProjects/TokYyStar/test2.png' ):
    one_path = OTHERS_DIR / ( get_new_file_number( OTHERS_DIR ) + "_one_" + suffix )

    img = Image.open( image_path ).convert( 'RGB' )

    transform = T.Compose([
        T.Resize( image_size ),  
        T.ToTensor(),
    ])

    input_tensor = transform( img ).unsqueeze( 0 )

    print( input_tensor.shape )

    with torch.no_grad():
        input_tensor = input_tensor.to( next( model.parameters() ).device ) 
        output = model( input_tensor ) 

    depth = output.squeeze().cpu().numpy()
    depth_normalized = ( depth - depth.min() ) / ( depth.max() - depth.min() )

    plt.figure( figsize = ( 10, 4 ) )

    plt.subplot( 1, 2, 1 )
    plt.imshow( img )
    plt.title( 'Input Image' )
    plt.axis( 'off' )

    plt.subplot( 1, 2, 2 )
    plt.imshow( depth_normalized, cmap = 'plasma' )  
    plt.title( 'Predicted Depth' )
    plt.axis( 'off' )

    plt.tight_layout()
    plt.savefig( one_path )


def main():
    name_to_architecture = {
        "resunet" : ResUNet,
        "cbam" : ResCBAMUNet,
        "atrous" : AtrousResCBAMUNet
    }

    args = parse_test_args()

    checkpont_name = os.path.splitext( args.checkpoint_name )[ 0 ]

    model, checkpointer = load_model_and_checkpointer( name_to_architecture[ args.architecture ](), args.checkpoint_path )

    if model == None and args.show:
        show()
        return 0
    elif model == None and not args.show: 
        return 0

    actions : Dict[ callable, bool] = {
        #batch( model = model ) : args.batch,
        dataset : args.dataset,
        predict( suffix = checkpont_name, model = model ) : args.predict,
        graphs( suffix = checkpont_name, checkpointer = checkpointer ) : args.graphs,
        one( suffix = checkpont_name, model = model ) : args.one,
        show : args.show
    }

    for act, flag_set in actions.items():
        if flag_set:
            act()


if __name__ == '__main__':
    main()
