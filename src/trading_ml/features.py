"""Compatibilidad de importación; la implementación vive en ``domain``."""

from .domain.features import FEATURES, feature_frame, with_labels

__all__ = ["FEATURES", "feature_frame", "with_labels"]
