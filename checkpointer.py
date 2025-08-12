import torch

import os

from tokyy.utils import log_message, LogType
from tokyy.metrics import Metrics, Metric
from typing import Dict, Optional


class Checkpointer:
    def __init__( self, model : torch.nn.Module, optimizer : torch.optim.Optimizer, criterion : torch.nn.Module, scaler = None, scheduler = None ):

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.epoch = 0
        self.learning_rates= []

        self.losses = {
            'train' : [],
            'val' : [],
            'test' : []
        }

        self._losses = {
            'train' : {},
            'val' : {},
            'test' : {}
        }

        for _loss in criterion._losses:
            self._losses[ 'train' ][ _loss ] = []
            self._losses[ 'val' ][ _loss ] = []
            self._losses[ 'test' ][ _loss ] = []

        self.metrics = {}
        for metric in Metrics:
            self.metrics[ metric ] = []


    def save( self, path : str ):
        """
        Saves the instance's: model, optimizer, scaler, epoch, lossses, _losses, metrics, learning_rates

        Args:
            path ( str ): Path to save file (e.g., dir/subdir/checkpoint.pt )
        
        Returns:
            No

        """

        checkpoint = {}

        checkpoint[ 'model' ] = self.model.state_dict()
        checkpoint[ 'optimizer' ] = self.optimizer.state_dict()

        if self.scaler:
            checkpoint[ 'scaler' ] = self.scaler.state_dict()

        if self.scheduler:
            checkpoint[ 'scheduler' ] = self.scheduler.state_dict()

        checkpoint[ 'epoch' ] = self.epoch
        checkpoint[ 'losses' ] = self.losses
        checkpoint[ '_losses' ] = self._losses
        checkpoint[ 'metrics' ] = self.metrics
        checkpoint[ 'learning_rates' ] = self.learning_rates

        try:             
            if os.path.exists( path ):
                log_message( LogType.WARNING, f"Overwriting to checkpoint saving path { path }" )
            else:
                os.makedirs( os.path.dirname( path ), exist_ok = True )
                log_message( LogType.WARNING, f"Created checkpoint saving path { path }" )

            torch.save( checkpoint, path )
            return log_message( LogType.SUCCESS, f"Checkpoint saved at epoch { self.epoch + 1 } to { path }" )
        
        except: 
            return log_message( LogType.ERROR, f"Checkpoint failed saving ", exit = False )


    def update( 
        self, 
        model : torch.nn.Module, 
        optimizer : torch.optim.Optimizer, 
        epoch : int, 
        scaler  = None, 
        scheduler = None, 
        losses : Optional[ Dict[ str, float ] ] = None,
        metrics : Optional[ Dict[ Metrics, float ] ] = None,
        _losses : Optional[ Dict[ str, Dict[ str, float ] ] ] = None, 
        learning_rate : float = 0.0 
    ):
        """
        Updates the checkpointer with new values.

        Args:
            model ( torch.nn.Module ) : The model being trained.
            optimizer ( torch.optim.Optimizer ) : The optimizer used for training.
            epoch ( int ) : The last completed training epoch.
            scaler  : Gradient scaler.
            scheduler  : Learning rate scheduler.
            losses ( Dict[ str, float ], optional ) : Dictionary with keys 'train', 'val', and 'test' holding float values.
            metrics ( Dict[ Metrics, float ], optional ) : Dictionary of evaluation metrics.
            _losses ( Dict[ str, Dict[ str, float ] ], optional ) : Dictionary with keys 'train', 'val', and 'test', each mapping to a sub-dictionary of loss values.
            learning_rate ( float ) : The current learning rate.

        Returns:
            No
        """
            
        self.model = model
        self.optimizer = optimizer

        if scaler:
            self.scaler = scaler
        
        if scheduler:
            self.scheduler = scheduler

        self.epoch = epoch
        self.learning_rates.append( learning_rate )

        for key, val in losses.items():
            if key not in self.losses:
                log_message( LogType.WARNING, f"Provided loss key { key } is not accepted. Skipping..." )
                continue
            
            self.losses[ key ].append( val )

        for _losses_set, _losses_dict in _losses.items():
            if _losses_set not in self._losses:
                log_message( LogType.WARNING, f"Provided _loss key { _losses_set } is not accepted. Skipping..." )
                continue
            
            for key, val in _losses_dict.items():
                if key not in self._losses[ _losses_set ]:
                    log_message( LogType.WARNING, f"Provided _loss key { key } is not accepted. Skipping..." )
                    continue
        
                self._losses[ _losses_set ][ key ].append( val )
            
        for key, val in metrics.items():
            if key not in self.metrics:
                log_message( LogType.WARNING, f"Provided metric key { key } is not accepted. Skipping..." )
                continue
            
            self.metrics[ key ].append( val )

        log_message( LogType.SUCCESS, "Checkpoint updated" )

    @classmethod
    def load( 
        cls, 
        path : str , 
        model : torch.nn.Module, 
        optimizer : torch.optim.Optimizer, 
        criterion : torch.nn.Module,
        scaler = None, 
        scheduler = None, 
        map_loc = 'cuda' 
        ):

        """
        Classmethod. 
        Loads checkpointer from path.

        Args:
            path ( str ) : Path to checkpoint file.
            model ( torch.nn.Module ) : Model architecture. Must be same with the one inisde the checkpoint.
            optimize ( torch.nn.Optimizer ) : Optimizer used to train saved model. Must be same type with the one inside the checkpoint.
            scaler : Gradient scaler used to train saved model. Must be same type with the one inside the checkpoint.
            scheduler : Learning rate scheduler. Must be same type with the one inside the checkpoint.
            map_loc ( str ): Where to load the tensors and map them to - 'cpu' or 'cuda'. Defaults to 'cuda'.
        """

        if not os.path.exists(path):
            log_message( LogType.ERROR, "Checkpoint load path doesn't exist.", exit = True )
            return None
    
        checkpoint = torch.load( path, map_location = map_loc, weights_only = False)
        
        instance = cls( model, optimizer, criterion, scaler, scheduler )

        if 'model' in checkpoint and model:
            instance.model.load_state_dict( checkpoint[ 'model' ] )
        else:
            log_message( LogType.ERROR, "Model not found in checkpoint", exit = True )
            return None

        if 'optimizer' in checkpoint and optimizer:
            instance.optimizer.load_state_dict( checkpoint[ 'optimizer' ] )
        else:
            log_message( LogType.WARNING, "Optimizer not found in checkpoint" )
        
        if scaler and 'scaler' in checkpoint:
            instance.scaler.load_state_dict( checkpoint[ 'scaler' ] )
        else:
            log_message( LogType.WARNING, "Scaler not found in checkpoint" )
        
        if scheduler and 'scheduler' in checkpoint:
            instance.scheduler.load_state_dict( checkpoint[ 'scheduler' ] )
        else:
            log_message( LogType.WARNING, "Scheduler not found in checkpoint" )
        
        if 'learning_rates' in checkpoint:
            instance.learning_rates = checkpoint[ 'learning_rates' ]
        else: 
            log_message( LogType.WARNING, "Scheduler learning rates not found in checkpoint" )

        if 'losses' in checkpoint:
            instance.losses = checkpoint[ 'losses' ]
        else:
            log_message( LogType.WARNING, "Losses not found in checkpoint" )

        if '_losses' in checkpoint:
            instance._losses = checkpoint[ '_losses' ]
        else:
            log_message( LogType.WARNING, "_Losses not found in checkpoint" )
        
        if 'metrics' in checkpoint:
            instance.metrics = checkpoint[ 'metrics' ]
        else:
            log_message( LogType.WARNING, "Metrics not found in checkpoint" )

        log_message( LogType.SUCCESS, "Checkpoint loaded" )

        return instance
    

    def show_metrics( self, last_n = 5 ):
        print( self.metrics.items() )
        for key, val in self.metrics.items():
            if len( val ) > 0:
                print( key.value, end = ' ' )
                print( val[ - min( len( val ), last_n ) ] )


    def show_losses( self, last_n = 5   ):
        for key, val in self.losses.items():
            if len( val ) > 0:
                print( key, end = ' ' )
                print( val[ - min( len( val ), last_n ) ] )

        for _losses_set, _losses_dict in self._losses.items():
            for key, val in _losses_dict.items():
                print( _losses_set, end = '_' )
                if len( val ) > 0:
                    print( key, end = ' ' )
                    print( val[ - min( len( val ), last_n ) ] )

    def show_learning_rates( self, last_n = 5 ):
        print( self.learning_rates[ -last_n ] )
