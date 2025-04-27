import torch.nn as nn

class CrossEntropyLossIgnoreBase(nn.CrossEntropyLoss):
    def __init__(self):
        # Assuming 12 is the correct index to ignore based on your context
        super().__init__(ignore_index=12) 

    def forward(self, input, target):
        # nnU-Net typically provides input as (batch_size, channels, x, y, z)
        # and target as (batch_size, 1, x, y, z)
        # CrossEntropyLoss expects input (N, C, ...) and target (N, ...)
        
        # input[0] might not be correct depending on how nnU-Net passes data.
        # Let's assume standard nnU-Net data format for now.
        # input shape: (batch_size, num_classes, d, h, w)
        # target shape: (batch_size, 1, d, h, w)
        
        input_tensor = input 
        # Squeeze the channel dimension from the target tensor
        target_tensor = target.squeeze(1) 
        
        # Call the superclass forward function with modified target
        return super().forward(input_tensor, target_tensor.long())
