from __future__ import annotations
from typing import Dict, Type
from .base import EmbeddingStrategy

_REGISTRY: Dict[str, Type[EmbeddingStrategy]] = {}


def register(cls: Type[EmbeddingStrategy]):
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **kwargs) -> EmbeddingStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_strategies() -> list:
    return sorted(_REGISTRY.keys())


# Import concrete strategies to register them
def _load():
    from . import hybrid  # noqa: F401
    try:
        from backend.comparison.external_models.paper4_lsb_magicmatrix import strategy as _p4  # noqa: F401
    except Exception:
        pass
    try:
        from backend.comparison.external_models.paper1_joint_cnn import strategy as _p1  # noqa: F401
    except Exception:
        pass
    try:
        from backend.comparison.external_models.paper3_block_prep_net import strategy as _p3  # noqa: F401
    except Exception:
        pass
    try:
        from backend.comparison.external_models.paper2_cyclegan_steg import strategy as _p2  # noqa: F401
    except Exception:
        pass
    try:
        from backend.comparison import ablation as _abl  # noqa: F401
    except Exception:
        pass

_load()
