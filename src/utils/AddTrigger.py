import torch
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).type(dtype=torch.float32)
    
class AddDatasetTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, weight):
        super(AddDatasetTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            raise ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    
    def _add_trigger(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
            img = self.add_trigger(img)
            img = img.squeeze()
        else:
            img = self.add_trigger(img)
        return img


    def __call__(self, img):
        """Get the poisoned image.
        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (C, H, W) or (H, W).

        Returns:
            torch.Tensor: The poisoned image.
        """
        
        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = self._add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            img = torch.from_numpy(img)
            img = self._add_trigger(img)
            img = img.numpy()
            return img
        elif type(img) == torch.Tensor:
            img = self._add_trigger(img)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))
    


    
class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = torch.tensor(y_target, dtype=torch.long)
        

    def __call__(self, y_target):
        return self.y_target