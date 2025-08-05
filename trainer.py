from tokyy.utils import LogType, log_message, ask_yes_no
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metric, Metrics

import torch, gc
from torch.utils.data import DataLoader
from torch.amp import autocast

import torchvision.transforms as T

import random

from PIL import Image

import os

from tqdm import tqdm

from typing import List

import sys


class Trainer():
    ask_before = True
    device = torch.device( 'cpu' )
    non_blocking = True
    clear_cache = True
    augment_data = True

    def __init__( self, model : torch.nn, optimizer : torch.optim, criterion : torch.nn, metrics : List[ Metrics ], checkpoint_path : str, scaler = None, scheduler = None ):
        self.model = model.to( Trainer.device )
        self.optimizer = optimizer

        self.scaler = scaler

        if type( scheduler ) == torch.optim.lr_scheduler.OneCycleLR:
            log_message( LogType.ERROR, "One Cycle LR scheduler can only be assigned after calling load_dataset method" )
            sys.exit( 1 )

        self.scheduler = scheduler

        self.criterion = criterion.to( Trainer.device )
        self.metric = Metric( metrics = metrics, device = Trainer.device )
        
        self.checkpoint_path = checkpoint_path
        self.checkpointer = Checkpointer.load( model = self.model, optimizer = self.optimizer, scaler = self.scaler, scheduler =self.scheduler, path = self.checkpoint_path ) if os.path.exists( checkpoint_path ) else Checkpointer( model, optimizer, scaler = scaler, scheduler = scheduler )
        
        self.num_workers = 8

        self.dataset = None
        self.train_loader = None
        self.test_loader = None

        self.accum_steps = 2
        self.batch_size = 32
        self.max_epochs = 100
        self.epochs_per_session = 20
        self.scheduler_steps_per_epoch = 0

        log_message( LogType.OK, f"Using device: { self.device }" )

        if self.clear_cache:
            Trainer.clear_cache()
            log_message( LogType.OK, "Cache cleared. To turn cache clearance off set Trainer.clear_cache to False." )

    def set_device( self, device_name = 'cpu' ):
        if device_name == 'cpu':
            Trainer.device = torch.device( 'cpu' )

            return log_message( LogType.SUCCESS, "Trainer's device changed to CPU" )

        elif device_name == 'cuda':
            if torch.cuda.is_available():
                Trainer.device = torch.device( 'cuda' )

                self.model = self.model.to( Trainer.device )
                self.criterion = self.criterion.to( Trainer.device )
                self.metric.device = Trainer.device

                log_message( LogType.SUCCESS, "Trainer's device changed to CUDA" )
                log_message( LogType.OK, f"Current device number: { torch.cuda.current_device() }" )
                log_message( LogType.OK, f"Current device name: { torch.cuda.get_device_name() }" )
                return LogType.SUCCESS.value            
            
            else:
                return log_message( LogType.WARNING, "CUDA is not available. Device remains CPU" )
    
        else:
            return log_message( LogType.WARNING, f"Provided device name not \'cpu\' or \'cuda\'. Device remains { self.device } ")


    def set_model( self, model ):
        self.model = model.to( self.device )


    def set_criterion( self, criterion ):
        self.criterion = criterion.to( self.device )


    def load_dataset( self, train_dataset, test_dataset  ):
        can_begin = ask_yes_no( "Load dataset?" ) if Trainer.ask_before else True

        if not can_begin:
            return log_message( LogType.NONE, "User input didn't allow dataset loading. Quittting...")

        log_message( LogType.OK, "Dataset started loading")

        torch.manual_seed( 69 )
        
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all( 69 )

        log_message( LogType.SUCCESS, "Datasets loaded" )

        self.train_loader = DataLoader( train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, pin_memory = True )
        self.test_loader = DataLoader( test_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, pin_memory = True )

        self.scheduler_steps_per_epoch = len( self.train_loader ) // self.accum_steps

        log_message( LogType.OK, f"Loaded { len( self.train_loader.dataset ) } training samples" )
        log_message( LogType.OK, f"Loaded { len( self.test_loader.dataset ) } training samples" )

        log_message( LogType.NONE, f"With a batch size of { self.batch_size } ==> { len( self.train_loader ) } batches for training" )
        log_message( LogType.NONE, f"With a batch size of { self.batch_size } ==> { len( self.test_loader ) } batches for testing" )


    def train_supervised( self ):
        can_begin = ask_yes_no( "Begin training?" ) if Trainer.ask_before else True

        if not can_begin:
            return log_message( LogType.NONE, "User input didn't allow training. Quittting...")

        if self.train_loader is None:
            return log_message( LogType.ERROR, "Training loader is empty. Quitting... " )

        start_epoch = self.checkpointer.get_epoch( self.checkpoint_path )

        for epoch in range( start_epoch, min( start_epoch + self.epochs_per_session, self.max_epochs ) ):
            self.model.train()

            train_loss = torch.tensor( 0.0, device = self.device )
            learning_rate = 0
            self.optimizer.zero_grad()

            loop = tqdm( enumerate( self.train_loader ), total = len( self.train_loader ), desc = f"Epoch { epoch + 1 } / { self.max_epochs }" )

            for i, ( input_data, target ) in loop:
                input_data = input_data.to( self.device, non_blocking = Trainer.non_blocking )
                target = target.to( self.device, non_blocking = Trainer.non_blocking )

                with autocast( device_type = self.device.type ):
                    output_data = self.model( input_data )
                    loss = self.criterion( output_data, target ) / self.accum_steps

                if torch.isfinite( loss) :
                    self.scaler.scale( loss ).backward()
                else:
                    log_message( LogType.WARNING, f"Skipping batch { i } due to invalid loss: { loss.item() }" )
                    continue

                if ( i + 1 ) % self.accum_steps == 0 or ( i + 1 ) == len( self.train_loader ):
                    self.scaler.step( self.optimizer )
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()

                train_loss += loss.detach() * input_data.size( 0 )

                loop.set_postfix( loss = loss.item() )

                learning_rate += self.optimizer.param_groups[ 0 ][ 'lr' ]

            log_message( LogType.OK, "Training completed. Evaluating on validation set..." )

            self.model.eval()

            val_loss = torch.tensor( 0.0, device = self.device )

            with torch.no_grad():
                for input_data, target in self.test_loader:
                    input_data = input_data.to( self.device, non_blocking = Trainer.non_blocking )
                    target = target.to( self.device, non_blocking = Trainer.non_blocking )

                    with autocast( device_type = self.device.type ):
                        output_data = self.model( input_data )
                        loss = self.criterion( output_data, target )

                        self.metric.compute_metrics( output_data, target, batch_size = self.batch_size )

                        val_loss += loss.detach() * input_data.size( 0 )

            train_loss = train_loss.item()
            val_loss = val_loss.item()

            losses = {
                'train_loss' : train_loss / len( self.train_loader.dataset ),
                'val_loss' : val_loss / len( self.test_loader.dataset )
            }

            log_message( LogType.NONE, f"Epoch [ {epoch + 1} / { self.max_epochs } ], Train Loss: {losses[ 'train_loss' ]:.4f}, Val Loss: {losses[ 'val_loss']:.4f}")

            self.checkpointer.update(
                model = self.model,
                optimizer = self.optimizer,
                scaler = self.scaler,
                scheduler = self.scheduler,

                epoch = epoch,
                learning_rate = learning_rate / len( self.train_loader.dataset ),

                losses = losses,
                metrics = self.metric.computed
            )

            self.checkpointer.save( self.checkpoint_path )

            log_message( LogType.SUCCESS, f"Model trained for { epoch + 1 } epoch[s]" )


    @staticmethod
    def clear_cache():
        gc.collect()
        torch.cuda.empty_cache()

    
    def data_augmentation( self ):
        pass
    