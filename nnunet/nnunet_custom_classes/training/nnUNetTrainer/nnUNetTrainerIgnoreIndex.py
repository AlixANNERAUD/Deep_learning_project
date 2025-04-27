from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# Adjust the import path based on your actual file structure if ce_ignore.py is placed differently
from Deep_learning_project.nnunet.nnUNet.nnunetv2.training.loss.ce_ignore import CrossEntropyLossIgnoreBase 

class nnUNetTrainerIgnoreIndex(nnUNetTrainer):
    def _build_loss(self):
        # Instantiate the correct loss class name
        loss = CrossEntropyLossIgnoreBase() 
        return loss
