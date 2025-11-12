"""
Cellpose-Gaussian: A Python package for Cellpose with Gaussian processing.

This package provides tools for cell segmentation with Gaussian-based processing.
"""

__version__ = "0.1.0"
__author__ = "Jimmy Lee"
__license__ = "MIT"

# Package exports for cellpose_gaussian
from .image_utils import (
    get_image_CHW,
    _to_numpy_hwC,
    _add_layer_from_hwC,
    reduce_resolution_same_size_ic,
    blur_and_boost_contrast,
    extract_arr_to_df,
    plot_segmentation_overlay,
)

__all__ = [
    "get_image_CHW",
    "_to_numpy_hwC",
    "_add_layer_from_hwC",
    "reduce_resolution_same_size_ic",
    "blur_and_boost_contrast",
    "extract_arr_to_df",
    "plot_segmentation_overlay",
]
