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

def edge_aware_smoothness_loss( depth, image ):
    depth_grad_x = gradient_x( depth )
    depth_grad_y = gradient_y( depth )
    image_grad_x = gradient_x( image ).abs().mean( 1, keepdim = True )
    image_grad_y = gradient_y( image ).abs().mean( 1, keepdim = True )
    
    smoothness_x = depth_grad_x * torch.exp( -image_grad_x )
    smoothness_y = depth_grad_y * torch.exp( -image_grad_y )
    
    return smoothness_x.abs().mean() + smoothness_y.abs().mean()


def scale_invariant_loss(pred, target, mask=None):
    if mask is not None:
        pred = pred[ mask ]
        target = target[ mask ]

    diff = torch.log( pred ) - torch.log( target )
    n = diff.numel()

    term1 = ( diff ** 2 ).sum() / n
    term2 = ( diff.sum() ** 2 ) / ( n ** 2 )
    loss = term1 - term2
    return loss


class DepthLoss( nn.Module ):
    def __init__( self, lambda_depth = 0.1, lambda_ssim = 1.0, kernel_size = 5 ):
        super().__init__()

        self.lambda_depth = lambda_depth
        self.lambda_ssim = lambda_ssim
        self.ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = kernel_size )

    def forward( self, pred, target ):
        l_depth = depth_loss( pred, target )
        l_grad = gradient_loss( pred, target )
        l_ssim = ssim_loss( pred, target, self.ssim )

        # print( f"l_depth: {l_depth:.4f} | {self.lambda_depth * l_depth:.4f}" )
        # print( f"l_grad: {l_grad:.4f}" )
        # print( f"l_ssim: {l_ssim:.4f} | {self.lambda_ssim * l_ssim:.4f}" )

        return self.lambda_depth * l_depth + l_grad + self.lambda_ssim * l_ssim


class DepthLossV2( nn.Module ):
    def __init__( self, lambda_berhu = 1.0, lambda_ews = 1.0, lambda_si = 1.0, lambda_ssim = 1.0, kernel_size = 5 ):
        self.lambda_berhu = lambda_berhu
        self.lambda_ews = lambda_ews
        self.lambda_si = lambda_si
        self.ssim = StructuralSimilarityIndexMeasure( data_range = 1.0, kernel_size = kernel_size )

    def forward( self, image, pred, target ):
        l_berhu = berhu_loss( pred, target )
        l_ews = edge_aware_smoothness_loss( pred, image )
        l_si = scale_invariant_loss( pred, target )
        l_ssim = ssim_loss( pred, target, self.ssim )

        return self.lambda_berhu * l_berhu + self.lambda_ews * l_ews + self.lambda_si * l_si + self.lambda_ssim  * l_ssim
    