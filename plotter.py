from tokyy.utils import LogType, log_message, get_new_file_number
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metrics
from tokyy import  LOSSES_DIR, METRICS_DIR, LEARNING_RATES_DIR, _LOSSES_TRAIN_DIR, _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, OTHERS_DIR, PREDICTS_DIR

import numpy as np

import torch
from torchvision import transforms as T

import h5py

import matplotlib.pyplot as plt

from PIL import Image



class Plotter():
    def __init__( self, model : torch.nn.Module, checkpointer : Checkpointer ):
        self.model = model
        self.metrics = checkpointer.metrics
        self.losses = checkpointer.losses
        self._losses = checkpointer._losses
        self.learning_rates = checkpointer.learning_rates
        self.sets = [ 'train', 'val', 'test' ]


    def plot_metrics( self, suffix : str ):
        metrics_path =  METRICS_DIR / ( get_new_file_number( METRICS_DIR ) + "_metrics_" + suffix)

        plt.figure( figsize = ( 12, 6 ) )

        for metric, values in self.metrics.items():
            if isinstance( metric, Metrics ):
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


    def plot_losses( self, suffix : str ):
        losses_path = LOSSES_DIR / ( get_new_file_number( LOSSES_DIR ) + "_losses_" + suffix )

        plt.figure( figsize = ( 12, 6 ) )

        for loss_type, values in self.losses.items():
            cpu_values = [
                x.detach().cpu().item() if torch.is_tensor(x) else float(x)
                for x in values
            ]

            plt.plot( cpu_values, label = loss_type )

        plt.title( "Training and Validation Loss Over Epochs" )
        plt.xlabel( "Epoch" )
        plt.ylabel( "Loss" )
        plt.legend()
        plt.grid( True )
        plt.tight_layout()
        plt.savefig( losses_path )
        plt.close()

        log_message( LogType.SUCCESS, f"Losses graph saved to { losses_path }" )


    def plot__losses( self, suffix : str ):
        set_to_path = {
            "train" : _LOSSES_TRAIN_DIR / ( get_new_file_number( _LOSSES_TRAIN_DIR ) + "__losses_train_" + suffix ),
            "test" :  _LOSSES_TEST_DIR / ( get_new_file_number( _LOSSES_TEST_DIR ) + "__losses_test_" + suffix ),
            "val" : _LOSSES_VAL_DIR / ( get_new_file_number( _LOSSES_VAL_DIR ) + "__losses_val_" + suffix )
        }

        for set in self.sets:
            plt.figure( figsize = ( 12, 6 ) )

            for _losses_set, _losses_dict in self._losses.items():
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


    def plot_learning_rates( self, suffix : str ):
        learning_rates_path = LEARNING_RATES_DIR / ( get_new_file_number( LEARNING_RATES_DIR ) + "_lr_" + suffix )

        plt.figure( figsize = ( 12, 6 ) )

        cpu_values = [
            x.detach().cpu().item() if torch.is_tensor( x ) else float( x )
            for x in self.learning_rates
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

    def predict( self, suffix : str ):
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
            T.Resize( ( 128, 128 ) ),
        ] )

        depth_transform = T.Compose( [
            T.ToTensor(),
            T.Resize( ( 128, 128 ) ),
        ] )

        n = len( file_paths )
        _, axs = plt.subplots( n, 3, figsize = ( 12, 4 * n ) )
        
        for i, file_path in enumerate( file_paths ):            
            with h5py.File( file_path, 'r' ) as f:
                rgb = np.array( f[ 'rgb' ] )
                depth = np.array( f[ 'depth' ] )

            if rgb.shape[ 0 ] == 3:
                rgb = np.transpose( rgb, ( 1, 2, 0 ) )

            rgb_tensor = rgb_transform( rgb ).unsqueeze( 0 ).to( torch.device( 'cuda' ) )
            depth_tensor = depth_transform( depth ).squeeze().cpu().numpy() / 10.0

            with torch.no_grad():
                pred = self.model( rgb_tensor )
                pred = pred.squeeze().cpu().numpy() 

            axs[ i ][ 0 ].imshow( rgb.astype( np.uint8 ) )
            axs[ i ][ 0 ].set_title( f"RGB Image [{ i }]" )
            axs[ i ][ 0 ].axis( "off" )

            axs[ i ][ 1 ].imshow( depth_tensor, cmap = 'plasma' )
            axs[ i ][ 1 ].set_title( "Ground Truth Depth" )
            axs[ i ][ 1 ].axis( "off" )

            axs[ i ][ 2 ].imshow( pred, cmap = 'plasma' )
            axs[ i ][ 2 ].set_title( "Predicted Depth" )
            axs[ i ][ 2 ].axis( "off" )

        plt.tight_layout()
        plt.savefig( predicts_path )

        log_message( LogType.OK, f"Preictions saved at { predicts_path }" )


    def predict_one( suffix : str, model : torch.nn.Module, image_path = r'/home/TokYy/PyTorchProjects/TokYyStar/test2.png' ):
        one_path = OTHERS_DIR / ( get_new_file_number( OTHERS_DIR ) + "_one_" + suffix )

        img = Image.open( image_path ).convert( 'RGB' )

        transform = T.Compose([
            T.Resize( ( 128, 128 ) ),  
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


    @classmethod
    def delete_by_nubmer( cls, number : str ):
        n_found = 0
        dirs = [ _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, _LOSSES_TRAIN_DIR, LEARNING_RATES_DIR, LOSSES_DIR, METRICS_DIR, OTHERS_DIR, PREDICTS_DIR ]

        for dir in dirs:
            for file in dir.iterdir():
                if file.is_file() and file.name[ : 2 ] == number:
                    file.unlink()
                    n_found +=1

                    log_message( LogType.SUCCESS, f"Delteted file { file.name }" )

        log_message( LogType.NONE, f"Deleted { n_found } files" )


    @classmethod 
    def delete_by_suffix( cls, suffix : str ):
        n_found = 0
        dirs = [ _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, _LOSSES_TRAIN_DIR, LEARNING_RATES_DIR, LOSSES_DIR, METRICS_DIR, OTHERS_DIR, PREDICTS_DIR ]

        for dir in dirs:
            for file in dir.iterdir():
                if file.is_file() and file.stem[ - len( suffix ) : ] == suffix:
                    file.unlink()
                    n_found += 1

                    log_message( LogType.SUCCESS, f"Delteted file { file.name } ")
        
        log_message( LogType.NONE, f"Deleted { n_found } files" )