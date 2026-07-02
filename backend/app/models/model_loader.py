from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from ml.src.models import create_model


class ModelUnavailableError(RuntimeError):
    """Raised when checkpoint inference is configured but cannot be used."""


@dataclass(slots=True)
class LoadedModel:
    model: nn.Module
    model_name: str
    class_names: list[str]
    input_size: int
    version: str
    is_smoke_test: bool
    checkpoint_path: Path


@lru_cache(maxsize=4)
def _load_cached(model_path_string: str, device_string: str) -> LoadedModel:
    model_path = Path(model_path_string)
    device = torch.device(device_string)
    if not model_path.exists():
        raise ModelUnavailableError(f"Model checkpoint not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        required = {"model_name", "model_state_dict", "class_names", "input_size"}
        missing = required.difference(checkpoint)
        if missing:
            raise ModelUnavailableError(f"Checkpoint is missing fields: {sorted(missing)}")
        model = create_model(
            str(checkpoint["model_name"]),
            len(checkpoint["class_names"]),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"Could not load checkpoint: {exc}") from exc
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        model_name=str(checkpoint["model_name"]),
        class_names=list(checkpoint["class_names"]),
        input_size=int(checkpoint["input_size"]),
        version=f"{checkpoint['model_name']}-epoch-{checkpoint.get('epoch', 'unknown')}",
        is_smoke_test=bool(checkpoint.get("is_smoke_test", False)),
        checkpoint_path=model_path,
    )


def load_model(model_path: Path, device: torch.device, allow_smoke_model: bool = False) -> LoadedModel:
    loaded = _load_cached(str(model_path.resolve()), str(device))
    if loaded.is_smoke_test and not allow_smoke_model:
        raise ModelUnavailableError(
            "The configured checkpoint is a smoke-test model. Set DERMASCAN_ALLOW_SMOKE_MODEL=true "
            "only for development verification."
        )
    return loaded
