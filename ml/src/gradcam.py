from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def find_last_convolution(model: nn.Module) -> nn.Conv2d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("Grad-CAM requires a model with at least one Conv2d layer.")
    return layers[-1]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer or find_last_convolution(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._capture_activations)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def _capture_gradients(
        self,
        _module: nn.Module,
        _gradient_input: tuple,
        gradient_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = gradient_output[0]

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_index: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects a single-image tensor with shape [1, C, H, W].")
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        selected_class = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
        logits[0, selected_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum = heatmap.min()
        maximum = heatmap.max()
        heatmap = (heatmap - minimum) / (maximum - minimum).clamp_min(1e-8)
        return heatmap.detach(), selected_class

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
