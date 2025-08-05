from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch
import os
import h5py
import numpy as np
from tokyy.utils import LogType, log_message
import sys

class NyuDepthV2( Dataset ):
    def __init__( self, root_dir : str, transform ):
        self.samples = []
        self.transform = transform

        self.n_scenes = 0

        if not os.path.exists( root_dir ):
            log_message( LogType.ERROR, f"Dataset path { root_dir } does not exist. Qutting..." )
            sys.exit( 1 )

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
    

if __name__ == "__main__":
    dataset_path = r"/home/TokYy/PyTorchProjects/GPU_Depth_Sextimation/nyu/val"

    dataset = Dataset( root_dir = dataset_path, size = ( 128, 128 ) )

    print(f"Dataset length: { len( dataset ) }" )

    rgb, depth = dataset[0]

    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}, min: {rgb.min()}, max: {rgb.max()}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}")


    rgb_min, rgb_max = float('inf'), float('-inf')
    depth_min, depth_max = float('inf'), float('-inf')

    for i in range(len(dataset)):
        rgb, depth = dataset[i]
        
        rgb_min = min(rgb_min, rgb.min().item())
        rgb_max = max(rgb_max, rgb.max().item())
        
        depth_min = min(depth_min, depth.min().item())
        depth_max = max(depth_max, depth.max().item())

    print(f'RGB Min: {rgb_min}, RGB Max: {rgb_max}')
    print(f'Depth Min: {depth_min}, Depth Max: {depth_max}')
