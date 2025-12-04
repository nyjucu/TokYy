from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch
import os
import h5py
import numpy as np
from tokyy.utils import LogType, log_message
import sys
from tqdm import tqdm


class NyuDepthV2( Dataset ):
    def __init__( self, root_dir : str, transform ):
        self.samples = []
        self.transform = transform

        self.n_scenes = 0

        if not os.path.exists( root_dir ):
            log_message( LogType.ERROR, f"Dataset path { root_dir } does not exist. Qutting...", exit = True )

        has_subdirs = any( os.path.isdir( os.path.join( root_dir, d )) for d in os.listdir( root_dir ) )

        if has_subdirs:
            for scene in os.listdir( root_dir ):
                scene_dir = os.path.join( root_dir, scene )

                if not os.path.isdir( scene_dir ):
                    continue

                self.n_scenes += 1


                for fname in os.listdir( scene_dir ):
                    if fname.endswith( '.h5' ):
                        self.samples.append( os.path.join( scene_dir, fname ) )
        else:
            for fname in os.listdir( root_dir ):
                if fname.endswith( '.h5' ):
                    self.samples.append( os.path.join( root_dir, fname ) )


    def __len__( self ):
        return len( self.samples )


    def __getitem__( self, idx ):
        file_path = self.samples[ idx ]

        with h5py.File( file_path, 'r' ) as f:
            rgb = np.array( f[ 'rgb' ] )
            depth = np.array( f[ 'depth' ] )

        return self.transform( rgb, depth )
    

class Kitti( Dataset ):
    def __init__( self, depth_root_dir : str, rgb_root_dir : str, transform ):
        self.samples = []
        self.transform = transform

        if not os.path.exists( depth_root_dir ):
            log_message( LogType.ERROR, f"Dataset path { depth_root_dir } does not exist. Quitting...", exit = True, log = not os.path.exists( depth_root_dir ) )

        if not os.path.exists( rgb_root_dir ):
            log_message( LogType.ERROR, f"Dataset path { rgb_root_dir } does not exist. Quitting...", exit = True )

        depth_has_subdirs = any( os.path.isdir( os.path.join( depth_root_dir, d )) for d in os.listdir( depth_root_dir ) )
        rgb_has_subdirs = any( os.path.isdir( os.path.join( rgb_root_dir, d )) for d in os.listdir( rgb_root_dir ) )

        have_subdirs = depth_has_subdirs and rgb_has_subdirs

        if have_subdirs:
            n_files_not_found = 0

            for drive in os.listdir( depth_root_dir ):
                rgb_drive_path = os.path.join( rgb_root_dir, drive )
                depth_drive_path = os.path.join( depth_root_dir, drive )

                if os.path.exists( rgb_drive_path ):
                    log_message( LogType.OK, f"{ drive } found inside { rgb_root_dir }" )
                else:
                    n_files_not_found += 1
                    continue

                rgb_drive_path_to_03 = os.path.join( rgb_drive_path, "image_03", "data" )
                rgb_drive_path_to_02 = os.path.join( rgb_drive_path, "image_02", "data" )

                depth_drive_path_to_03 = os.path.join( depth_drive_path, "proj_depth", "groundtruth", "image_03" )
                depth_drive_path_to_02 = os.path.join( depth_drive_path, "proj_depth", "groundtruth", "image_02" )

                if not os.path.exists( rgb_drive_path_to_03 ): 
                    log_message( LogType.WARNING, f"{ rgb_drive_path_to_03 } not found. Skipping." )
                    continue
                
                if not os.path.exists( rgb_drive_path_to_02 ): 
                    log_message( LogType.WARNING, f"{ rgb_drive_path_to_02 } not found. Skipping." )
                    continue

                if not os.path.exists( depth_drive_path_to_03 ): 
                    log_message( LogType.WARNING, f"{ depth_drive_path_to_03 } not found. Skipping." )
                    continue

                if not os.path.exists( depth_drive_path_to_02 ): 
                    log_message( LogType.WARNING, f"{ depth_drive_path_to_02 } not found. Skipping." )
                    continue

                for image in os.listdir( depth_drive_path_to_03 ):
                    if image.endswith( ".png" ):
                        depth_image_path = os.path.join( depth_drive_path_to_03, image )
                        rgb_image_path = os.path.join( rgb_drive_path_to_03, image )

                        if not os.path.exists( rgb_image_path ):
                            log_message( LogType.WARNING, f"{ rgb_image_path } not found. Skipping." )
                            continue

                        self.samples.append( ( rgb_image_path, depth_image_path ) )

            log_message( LogType.WARNING, f"{ n_files_not_found } files not found inside { rgb_root_dir }" )


    def __len__( self ):
        return len( self.samples )


    def __getitem__( self, idx ):
        rgb_image_path, depth_image_path = self.samples[ idx ]

        depth = Image.open( depth_image_path )
        depth = np.array( depth )
        
        depth[ depth == 0.0 ] = 21000

        rgb = Image.open( rgb_image_path )
        rgb = np.array( rgb )

        if self.transform is None:
            return ( rgb, depth )
        
        return self.transform( rgb, depth )
        

def some_info( dataset  ):
    print( f"Dataset length: { len( dataset ) }" )

    rgb, depth = dataset[ 0 ]

    print( f"RGB shape: { rgb.shape }, dtype: { rgb.dtype }, min: { rgb.min() }, max: { rgb.max() }" )
    print( f"Depth shape: { depth.shape }, dtype: { depth.dtype }, min: { depth.min() }, max: { depth.max() }" )


    rgb_min, rgb_max = float( 'inf' ), float( '-inf' )
    depth_min, depth_max = float( 'inf' ), float( '-inf' )

    for i in tqdm( range( len( dataset ) ), desc = "Parsing dataset" ):
        rgb, depth = dataset[i]
        
        rgb_min = min( rgb_min, rgb.min().item() )
        rgb_max = max( rgb_max, rgb.max().item() )
        
        depth_min = min( depth_min, depth.min().item() )
        if depth_max < depth.max().item(): depth_max = depth.max().item()

    print( f'RGB Min: { rgb_min }, RGB Max: { rgb_max }' )
    print( f'Depth Min: { depth_min }, Depth Max: { depth_max }' )
