import numpy as np
import torch
import torchvision.transforms as T
import random
import cv2
from PIL import Image

from tokyy.utils import LogType, log_message


class Resize:
    def __init__( self, size ):
        self.size = size

    def __call__( self, rgb, depth ):
        if rgb.shape[ 0 ] == 3:
            rgb = np.transpose( rgb, ( 1, 2, 0 ) )

        rgb = cv2.resize( rgb, ( self.size[ 1 ], self.size[ 0 ] ), interpolation = cv2.INTER_LINEAR )
        depth = cv2.resize( depth, ( self.size[ 1 ], self.size[ 0 ]), interpolation = cv2.INTER_NEAREST )
        return rgb, depth


class RandomHorizontalFlip:
    def __init__( self, p = 0.5 ):
        self.p = p

    def __call__( self, rgb, depth ):
        if random.random() < self.p:
            rgb = np.ascontiguousarray( np.fliplr( rgb ) )
            depth = np.ascontiguousarray( np.fliplr( depth ) )
        return rgb, depth


class RandomCrop:
    def __init__( self, size ):
        self.size = size

    def __call__( self, rgb, depth ):
        h, w = rgb.shape[ : 2 ]
        th, tw = self.size
        if h < th or w < tw:
            raise log_message( LogType.ERROR, f"Crop size { self.size } is bigger than image size { ( h, w ) }" )

        i = random.randint( 0, h - th )
        j = random.randint( 0, w - tw )
        rgb = rgb[ i : i + th, j : j + tw, : ]
        depth = depth[ i : i + th, j : j + tw ]

        return rgb, depth


class RandomRotationByAngle:
    def __init__( self, angle_range = ( -20, 20 ) ):  
        self.angle_range = angle_range

    def __call__(self, rgb, depth):
        angle = random.uniform( self.angle_range[ 0 ], self.angle_range[ 1 ] )

        h, w = rgb.shape[ : 2 ]
        center = ( w // 2, h // 2 )

        M = cv2.getRotationMatrix2D( center, angle, scale = 1.0 )

        rgb_rot = cv2.warpAffine( rgb, M, ( w, h ), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101 )

        depth_rot = cv2.warpAffine( depth, M, ( w, h ), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_REFLECT_101 )

        return rgb_rot, depth_rot


class ToTensor:
    def __init__( self, rgb_normalize : float, depth_normalize : float ):
        self.rgb_normlaize = rgb_normalize
        self.depth_normalize = depth_normalize

    def __call__( self, rgb, depth, rgb_normalize, depth_normalize ):
        rgb = torch.from_numpy( rgb.transpose( 2, 0, 1 ) ).float() / self.rgb_normalize
        depth = torch.from_numpy( depth ).unsqueeze( 0 ).float() / self.depth_normalize
        return rgb, depth


class ColorJitter:
    def __init__( self, brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1 ):
        self.jitter = T.ColorJitter( brightness, contrast, saturation, hue )

    def __call__( self, rgb, depth ):
        rgb_pil = Image.fromarray( rgb )
        rgb_pil = self.jitter( rgb_pil )
        rgb = np.array( rgb_pil )
        return rgb, depth


class PairedCompose:
    def __init__( self, transforms ):
        self.transforms = transforms

    def __call__( self, rgb, depth ):
        for t in self.transforms:
            rgb, depth = t( rgb, depth )

        return rgb, depth
