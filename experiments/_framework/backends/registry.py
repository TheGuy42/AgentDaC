from enum import StrEnum
from typing import Callable
from importlib.util import find_spec

from experiments._framework.backends.backend import BackendArgs, Backend
from src.utils.logging import create_logger

logger = create_logger(__name__)


class BackendName(StrEnum):
    ART = "art"
    VERL = "verl"
    VLLM = "vllm"


REGISTRY: dict[BackendName, Callable[[BackendArgs], Backend]] = {}


def auto_backend() -> BackendName:

    for kind in BackendName:
        if find_spec(kind.value) is not None:
            logger.info(f"Detected {kind.value} backend.")
            return kind

    raise RuntimeError(f"No supported backend detected. Please install {[kind.value for kind in BackendName]}")


def register_backend(kind: BackendName) -> Callable:
    """Register a new backend class under the given kind."""

    def decorator(func: Callable[[BackendArgs], Backend]) -> Callable[[BackendArgs], Backend]:
        if kind in REGISTRY:
            raise ValueError(f"{kind!r} is already registered")

        REGISTRY[kind] = func
        return func

    return decorator


@register_backend(BackendName.ART)
def _build_art_backend(args: BackendArgs) -> Backend:
    from experiments._framework.backends.art import ArtBackend

    return ArtBackend(args)


@register_backend(BackendName.VERL)
def _build_verl_backend(args: BackendArgs) -> Backend:
    from experiments._framework.backends.verl import VerlBackend

    return VerlBackend(args)


@register_backend(BackendName.VLLM)
def _build_vllm_backend(args: BackendArgs) -> Backend:
    from experiments._framework.backends.vllm import VllmBackend

    return VllmBackend(args)


def create_backend(args: BackendArgs, kind: BackendName | None = None) -> Backend:
    """Create a backend instance of the given kind."""
    if kind is None:
        kind = auto_backend()

    if kind not in REGISTRY:
        raise ValueError(f"Unknown backend kind: {kind}")
    return REGISTRY[kind](args)
