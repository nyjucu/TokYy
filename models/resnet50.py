import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck( nn.Module ):
    def __init__( self, in_channels, out_channels, do_downsample = False, stride = 1, ratio = 4 ):
        super().__init__()

        self.do_downsample = do_downsample
        u_channels = out_channels // ratio

        self.conv1 = nn.Conv2d( in_channels, u_channels, kernel_size = 1, stride = 1, bias = False )
        self.bn1 = nn.BatchNorm2d( u_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True )
        self.conv2 = nn.Conv2d( u_channels, u_channels, kernel_size = 3, stride = stride, padding = 1, bias = False )
        self.bn2 = nn.BatchNorm2d( u_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True )
        self.conv3 = nn.Conv2d(u_channels, out_channels, kernel_size = 1, stride = 1, bias = False )
        self.bn3 = nn.BatchNorm2d( out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True )
        self.relu = nn.ReLU( inplace = True )

        self.downsample = None

        if do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d( in_channels, out_channels, kernel_size = 1, stride = stride, bias = False ),
                nn.BatchNorm2d( out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True )
            )


    def forward( self, x ):
        identity = x

        x = self.conv1( x )
        x = self.bn1( x )
        x = self.conv2( x )
        x = self.bn2( x )
        x = self.conv3( x )
        x = self.bn3( x )

        if self.do_downsample:
            identity = self.downsample( identity )

        x += identity
        x = self.relu(x)

        return x

    

class PyTorchResNet50( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.conv1 = nn.Conv2d( 3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False )
        self.bn1 = nn.BatchNorm2d( 64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True )
        self.relu = nn.ReLU( inplace = True )

        self.maxpool = nn.MaxPool2d( kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode = False )       
        
        self.layer1 = nn.Sequential(
            Bottleneck( 64, 256, do_downsample = True ),
            Bottleneck( 256, 256 ),
            Bottleneck( 256, 256 ),
        )

        self.layer2 = nn.Sequential(
            Bottleneck( 256, 512, do_downsample = True, stride = 2),
            Bottleneck( 512, 512 ),
            Bottleneck( 512, 512 ),
            Bottleneck( 512, 512 ),
        )

        self.layer3 = nn.Sequential(
            Bottleneck( 512, 1024, do_downsample = True, stride = 2),
            Bottleneck( 1024, 1024),
            Bottleneck( 1024, 1024),
            Bottleneck( 1024, 1024),
            Bottleneck( 1024, 1024),
            Bottleneck( 1024, 1024),
        )

        self.layer4 = nn.Sequential(
            Bottleneck( 1024, 2048, do_downsample = True, stride = 2),
            Bottleneck( 2048, 2048 ),
            Bottleneck( 2048, 2048 ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d( output_size = (1, 1) )
        self.fc = nn.Linear( in_features = 2048, out_features = 1000, bias = True )


    def forward( self, x ):
        x = self.conv1( x )
        x = self.bn1( x )
        x = self.relu( x )

        x = self.maxpool( x )

        x = self.layer1( x )
        x = self.layer2( x )
        x = self.layer3( x )
        x = self.layer4( x )

        x = self.avgpool( x )
        x = torch.flatten( x, 1 )
        x = self.fc( x )
