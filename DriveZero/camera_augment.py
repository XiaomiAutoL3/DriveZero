"""GridMask augmentation used by the visual encoder."""
import torch
from torch import nn

class GridMask(nn.Module):
    """Apply one shared GridMask pattern across a batch of camera images."""

    def __init__(self) -> None:
        super().__init__()
        self.probability = 0.7
        self.ratio = 0.5

    @torch.compiler.disable
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply one shared random mask to all samples, cameras, and channels."""
        if not self.training:
            return images
        if torch.rand((), device=images.device).item() > self.probability:
            return images
        if images.ndim != 4:
            raise ValueError("GridMask input must have shape [B*N, C, H, W].")

        _, _, height, width = images.shape
        if height <= 2:
            raise ValueError("GridMask input height must be greater than 2.")
        mask_height = int(1.5 * height)
        mask_width = int(1.5 * width)
        distance = int(torch.randint(2, height, (), device=images.device).item())
        length = min(max(int(distance * self.ratio + 0.5), 1), distance - 1)
        start_h = int(torch.randint(distance, (), device=images.device).item())
        start_w = int(torch.randint(distance, (), device=images.device).item())

        mask = torch.ones(
            (mask_height, mask_width), dtype=images.dtype, device=images.device
        )
        for index in range(mask_height // distance):
            start = distance * index + start_h
            mask[start : min(start + length, mask_height), :] = 0
        for index in range(mask_width // distance):
            start = distance * index + start_w
            mask[:, start : min(start + length, mask_width)] = 0

        top = (mask_height - height) // 2
        left = (mask_width - width) // 2
        mask = mask[top : top + height, left : left + width]
        mask = 1 - mask
        return images * mask.view(1, 1, height, width)
