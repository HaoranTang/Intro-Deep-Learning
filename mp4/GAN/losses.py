import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
import torch.nn.functional as F

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_loss = bce_loss(logits_real, torch.ones(logits_real.shape).cuda().detach(), reduction = 'mean')
    fake_loss = bce_loss(1 - logits_fake, torch.ones(logits_fake.shape).cuda().detach(), reduction = 'mean')
    loss=(real_loss+fake_loss)/2
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########
    
    return bce_loss(logits_fake, torch.ones(logits_fake.shape).cuda().detach(), reduction = 'mean')


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_loss = F.mse_loss(scores_real, torch.ones(scores_real.shape).cuda().detach(), reduction = 'mean')
    fake_loss = F.mse_loss(1 - scores_fake, torch.ones(scores_fake.shape).cuda().detach(), reduction = 'mean')
    loss=(real_loss+fake_loss)/2
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########
    
    return F.mse_loss(scores_fake, torch.ones(scores_fake.shape).cuda().detach(), reduction = 'mean')
