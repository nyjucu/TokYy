import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = 5 )


def depth_loss( pred, target ):
    return F.l1_loss( pred, target )


def gradient_loss( pred, target ):
    pred_dy = pred[ :, :, 1:, : ] - pred[ :, :, :-1, : ]
    pred_dx = pred[ :, :, :, 1: ] - pred[ :, :, :, :-1 ]

    target_dy = target[:, :, 1:, :] - target[ :, :, :-1, : ]
    target_dx = target[:, :, :, 1:] - target[ :, :, :, :-1 ]

    loss_dy = F.l1_loss( pred_dy, target_dy )
    loss_dx = F.l1_loss( pred_dx, target_dx )

    return loss_dx + loss_dy


def ssim_loss( pred, target, ssim : StructuralSimilarityIndexMeasure ):
    return 1 - ssim( pred, target )


def berhu_loss( pred, target ):
    error = torch.abs( pred - target )  
    
    threshold = 0.2 * torch.max( error ).item()

    l1_part = error[ error <= threshold ]
    l2_part = error[ error > threshold ]
    
    loss_l1 = l1_part
    loss_l2 = ( l2_part ** 2 + threshold ** 2 ) / ( 2 * threshold )
    
    loss = torch.cat( [ loss_l1, loss_l2 ] )
    return loss.mean()
        

def gradient_x( img ):
    return img[ :, :, :, :-1 ] - img[ :, :, :, 1: ]


def gradient_y(img):
    return img[ :, :, :-1, : ] - img[ :, :, 1:, : ]


def edge_aware_smoothness_loss( depth, pred ):
    depth_grad_x = gradient_x( depth )
    depth_grad_y = gradient_y( depth )
    pred_grad_x = gradient_x( pred ).abs().mean( 1, keepdim = True )
    pred_grad_y = gradient_y( pred ).abs().mean( 1, keepdim = True )
    
    smoothness_x = depth_grad_x * torch.exp( - pred_grad_x )
    smoothness_y = depth_grad_y * torch.exp( - pred_grad_y )
    
    return smoothness_x.abs().mean() + smoothness_y.abs().mean()


def scale_invariant_loss( pred, target, epsilon = 1e-6, mask = None ):
    if mask is not None:
        pred = pred[ mask ]
        target = target[ mask ]

    pred = torch.clamp( pred, min = epsilon )
    target = torch.clamp( target, min = epsilon )

    diff = torch.log( pred ) - torch.log( target )
    n = diff.numel()

    term1 = ( diff ** 2 ).sum() / n
    term2 = ( diff.sum() ** 2 ) / ( n ** 2 )
    loss = term1 - term2
    return loss


class DepthLoss( nn.Module ):
    def __init__( self, lambda_depth = 0.1, lambda_grad = 1.0, lambda_ssim = 1.0, kernel_size = 5 ):
        super().__init__()

        self.lambda_depth = lambda_depth
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        
        self.ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = kernel_size )

        self._losses = [ "depth", "gradient", "ssim" ]

    def forward( self, pred, target ):
        _losses = {
            "depth" : depth_loss( pred, target ) * self.lambda_depth,
            "gradient" : gradient_loss( pred, target ) * self.labda_grad,
            "ssim" : ssim_loss( pred, target, self.ssim ) * self.lambda_ssim
        }

        return sum( _losses.values() ), _losses


class DepthLossV2( nn.Module ):
    def __init__( self, lambda_berhu = 1.0, lambda_ews = 1.0, lambda_si = 1.0, lambda_ssim = 1.0, kernel_size = 5 ):
        super().__init__()

        self.lambda_berhu = lambda_berhu
        self.lambda_ews = lambda_ews
        self.lambda_si = lambda_si
        self.lambda_ssim = lambda_ssim

        self.ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = kernel_size )

        self._losses = [ "berhu", "edge_aware_smoothness", "scale_invariant", "ssim" ]

    def forward( self, pred, target ):
        _losses = {
            "berhu" : berhu_loss( pred, target ) * self.lambda_berhu,
            "edge_aware_smoothness" : edge_aware_smoothness_loss( pred, target ) * self.lambda_ews,
            "scale_invariant" : scale_invariant_loss( pred, target ) * self.lambda_si,
            "ssim" : ssim_loss( pred, target, self.ssim ) * self.lambda_ssim
        }

        print( f"berhu : {_losses[ "berhu" ]:.4f}" )
        print( f"edge_aware_smothness : {_losses[ "berhu" ]:.4f}" )
        print( f"scale_invariant : {_losses[ "scale_invariant" ]:.4f}" )
        print( f"ssim : {_losses[ "ssim" ]:.4f}" )

        for k, v in _losses.items():
            print( k, end = ' ' )
            print( v.detach().item() )

        print( sum( _losses.values() ).detach().item() )

        return sum( _losses.values() ), _losses


class DepthLossV3( nn.Module ):
    def __init__( self, lambda_berhu = 1.0, lambda_grad = 1.0, lambda_si = 1.0, lambda_ssim = 1.0, kernel_size = 5 ):
        super().__init__()

        self.lambda_berhu = lambda_berhu
        self.lambda_grad = lambda_grad
        self.lambda_si = lambda_si
        self.lambda_ssim = lambda_ssim

        self.ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = kernel_size )

        self._losses = [ "berhu", "gradient", "scale_invariant", "ssim" ]

    def forward( self, pred, target ):
        _losses = {
            "berhu" : berhu_loss( pred, target ) * self.lambda_berhu,
            "gradient" : gradient_loss( pred, target ) * self.lambda_grad,
            "scale_invariant" : scale_invariant_loss( pred, target ) * self.lambda_si,
            "ssim" : ssim_loss( pred, target, self.ssim ) * self.lambda_ssim
        }

        for k, v in _losses.items():
            print( k, end = ' ' )
            print( v.detach().item() )

        print( sum( _losses.values() ).detach().item() )

        return sum( _losses.values() ), _losses
