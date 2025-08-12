from tokyy.utils import LogType, log_message, get_new_file_number
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metrics
from tokyy import  LOSSES_DIR, METRICS_DIR, LEARNING_RATES_DIR, _LOSSES_TRAIN_DIR, _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, OTHERS_DIR, PREDICTS_DIR

import torch

import matplotlib.pyplot as plt


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

    @staticmethod
    def delete_by_nubmer( number : str ):
        n_found = 0
        dirs = [ _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, _LOSSES_TRAIN_DIR, LEARNING_RATES_DIR, LOSSES_DIR, METRICS_DIR, OTHERS_DIR, PREDICTS_DIR ]

        for dir in dirs:
            for file in dir.iterdir():
                if file.is_file() and file.name[ : 2 ] == number:
                    file.unlink()
                    n_found +=1

                    log_message( LogType.SUCCESS, f"Delteted file { file.name }" )

        log_message( LogType.NONE, f"Deleted { n_found } files" )

    @staticmethod 
    def delete_by_suffix( suffix : str ):
        n_found = 0
        dirs = [ _LOSSES_TEST_DIR, _LOSSES_VAL_DIR, _LOSSES_TRAIN_DIR, LEARNING_RATES_DIR, LOSSES_DIR, METRICS_DIR, OTHERS_DIR, PREDICTS_DIR ]

        for dir in dirs:
            for file in dir.iterdir():
                if file.is_file() and file.stem[ - len( suffix ) : ] == suffix:
                    file.unlink()
                    n_found += 1

                    log_message( LogType.SUCCESS, f"Delteted file { file.name } ")
        
        log_message( LogType.NONE, f"Deleted { n_found } files" )