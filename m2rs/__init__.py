"""M2RS: multi-mechanism robust segmentation for binary protocol field
boundary inference."""

from .engine import DEFAULT_CFG, M2RSBoundaryEngine

__version__ = '1.0.0'
__all__ = ['M2RSBoundaryEngine', 'DEFAULT_CFG', '__version__']
