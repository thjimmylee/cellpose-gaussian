#!/usr/bin/env python3
"""
Image handling utilities for the cellpose-gaussian package.

Contains helpers to normalize xarray / ImageContainer images into NumPy arrays,
apply simple filters / downsampling, and utilities for extracting region
statistics and plotting segmentation overlays.
"""
from typing import Optional, Tuple, Sequence, Any

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table, regionprops
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity


def get_image_CHW(
    img: xr.Dataset,
    layer: str = "image",
    library_id: Optional[Any] = None,
    scale: Optional[int] = None,
    compute: bool = True,
) -> np.ndarray:
    """
    Return an image layer as a NumPy ndarray with channels-first layout (C, H, W).

    This helper accepts an xarray-backed image container dataset (typical of
    Squidpy/Scanpy ImageContainer usage) and normalizes a selected layer into
    a NumPy array with shape (C, H, W). It is tolerant of various dim order
    conventions and will attempt to infer the channel axis if not explicitly
    named.

    Parameters
    ----------
    img
        xarray-like dataset containing the requested layer, e.g. an ImageContainer.
    layer
        Name of the layer to extract.
    library_id
        If the layer has a 'library_id' dimension, select this value. If None
        and the dim exists, the first value will be used.
    scale
        If the layer has a 'scale' (image pyramid) dimension, select this.
        If None and the dim exists, the smallest available scale index is chosen
        (commonly the highest-resolution).
    compute
        If True and the underlying data is dask-backed, compute to NumPy.

    Returns
    -------
    np.ndarray
        Array with shape (C, H, W). If the original was 2-D (H, W) it will be
        returned as (1, H, W).

    Raises
    ------
    ValueError
        If an unsupported layout is encountered.
    """
    if layer not in img:
        raise KeyError(f"Layer '{layer}' not found in provided dataset")

    da: xr.DataArray = img[layer]

    # Select library if present
    if "library_id" in da.dims:
        if library_id is None:
            library_id = da.coords["library_id"].values[0]
        da = da.sel(library_id=library_id)

    # Select scale (pyramid level) if present
    if "scale" in da.dims:
        if scale is None:
            # choose minimum scale coordinate (commonly highest-res)
            try:
                scale = int(da.coords["scale"].values.min())
            except Exception:
                scale = int(da.coords["scale"].values[0])
        da = da.sel(scale=scale)

    # Squeeze singleton dims except the canonical ones we want to keep
    # (y, x, channels) — this avoids accidentally keeping singletons like z or t.
    keep = {"y", "x", "channels"}
    for d in list(da.dims):
        if d not in keep and da.sizes.get(d, 1) == 1:
            da = da.isel({d: 0})

    # Materialize to NumPy
    if compute and hasattr(da.data, "compute"):
        arr = da.data.compute()
    else:
        arr = da.to_numpy()

    # Normalize to (C, H, W)
    dims = list(da.dims)

    if arr.ndim == 2:
        # (H, W) -> (1, H, W)
        return arr[None, ...]

    if arr.ndim == 3:
        # If channels are named explicitly, move them to axis 0
        if "channels" in dims:
            c_axis = dims.index("channels")
            if c_axis != 0:
                arr = np.moveaxis(arr, c_axis, 0)
            return arr  # already (C, H, W)
        # Otherwise infer: assume the smallest axis is channels, else treat last axis as channels
        c_axis = int(np.argmin(arr.shape))
        if arr.shape[c_axis] > 16:
            # heuristic fallback: treat last axis as channels if smallest axis is large
            c_axis = -1
        if c_axis != 0:
            arr = np.moveaxis(arr, c_axis, 0)
        return arr

    if arr.ndim == 4:
        # Try to collapse the first non-(y/x/channels) axis if present
        if "channels" in dims:
            # remove any extra singleton or indexable dims by taking the first index
            for i, d in enumerate(dims):
                if d not in ("y", "x", "channels"):
                    da = da.isel({d: 0})
            # re-materialize
            if compute and hasattr(da.data, "compute"):
                arr = da.data.compute()
            else:
                arr = da.to_numpy()
            dims = list(da.dims)
            c_axis = dims.index("channels")
            if c_axis != 0:
                arr = np.moveaxis(arr, c_axis, 0)
            return arr
        else:
            # fallback: assume shape (extra, H, W, C) -> take first on extra and proceed
            arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Unexpected 4D layout after slicing: shape={arr.shape}, dims={dims}")
            c_axis = int(np.argmin(arr.shape))
            if c_axis != 0:
                arr = np.moveaxis(arr, c_axis, 0)
            return arr

    raise ValueError(f"Unsupported image shape {arr.shape} with dims {dims}")


def _to_numpy_hwC(img: xr.Dataset, layer: str = "image") -> np.ndarray:
    """
    Return the requested layer as a channels-last NumPy array with shape (H, W, C).

    Behavior:
      - selects z=0 if a z dimension is present
      - materializes dask-backed arrays when present
      - moves a 'channels' named axis to the last position if necessary

    Parameters
    ----------
    img
        xarray dataset containing the layer.
    layer
        name of the layer to extract.

    Returns
    -------
    np.ndarray
        Array shaped (H, W, C) or (H, W, 1) if originally 2D.

    Raises
    ------
    ValueError
        If an unexpected number of dimensions is found.
    """
    if layer not in img:
        raise KeyError(f"Layer '{layer}' not found in provided dataset")
    da = img[layer]
    if "z" in da.dims:
        da = da.isel(z=0)

    arr = da.data.compute() if hasattr(da.data, "compute") else da.values
    dims = list(da.dims)

    # Ensure channels-last
    if "channels" in dims:
        c_ax = dims.index("channels")
        if c_ax != arr.ndim - 1:
            arr = np.moveaxis(arr, c_ax, -1)
    elif arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported array ndim {arr.ndim} from dims {dims}")
    return arr


def _add_layer_from_hwC(img: Any, arr: np.ndarray, layer_name: str) -> None:
    """
    Add a (H, W, C) NumPy array back into the image container as a new layer.

    The container must support a method `.add_img(array, layer=<name>, dims=(...))`
    as used by Squidpy's ImageContainer. If the object does not expose that
    method, a TypeError is raised.

    Parameters
    ----------
    img
        Image container-like object with add_img method.
    arr
        NumPy array shaped (H, W, C).
    layer_name
        Name of the new layer to add.

    Raises
    ------
    TypeError
        If the image container does not support add_img.
    """
    if not hasattr(img, "add_img"):
        raise TypeError("Provided image object does not support add_img(layer=...)")
    img.add_img(arr, layer=layer_name, dims=("y", "x", "channels"))


def reduce_resolution_same_size_ic(
    img: xr.Dataset,
    in_layer: str = "image",
    out_layer: str = "image_lowres",
    factor: float = 0.25,
    pre_blur_sigma: Optional[float] = None,
    grayscale_first: bool = False,
    gray_channel: int = 0,
) -> str:
    """
    Reduce apparent resolution by downsampling then upsampling while keeping image
    pixel dimensions (H, W) unchanged. Saves the result as a new layer in the
    provided image container.

    This is useful to synthesize lower resolution / blur-like degradations while
    preserving geometry for downstream testing.

    Parameters
    ----------
    img
        Image container (xarray dataset or similar) containing `in_layer`.
    in_layer
        Input layer name.
    out_layer
        Name for the created output layer.
    factor
        Downsample factor in each spatial dimension (0 < factor <= 1). Example:
        factor=0.25 will produce a roughly 4x lower resolution intermediate.
    pre_blur_sigma
        Optional Gaussian sigma applied before downsampling to reduce aliasing.
    grayscale_first
        If True, collapse channels to a single grayscale channel before resizing.
    gray_channel
        Index of the channel to keep if grayscale_first is True.

    Returns
    -------
    str
        Name of the added layer (same as out_layer).

    Notes
    -----
    - The function expects the container layer to be convertible to an (H, W, C)
      NumPy array via _to_numpy_hwC.
    - dtype and range are preserved for integer and floating types where possible.
    """
    arr = _to_numpy_hwC(img, in_layer)          # (H, W, C)
    H, W, C = arr.shape
    orig_dtype = arr.dtype

    if grayscale_first:
        arr = arr[..., gray_channel:gray_channel + 1]  # (H, W, 1)

    if pre_blur_sigma is not None and pre_blur_sigma > 0:
        # skimage.gaussian supports channel_axis for multichannel arrays
        arr = gaussian(arr, sigma=pre_blur_sigma, channel_axis=-1, preserve_range=True)

    # Downsample then upsample back to original size
    h_small = max(1, int(round(H * factor)))
    w_small = max(1, int(round(W * factor)))
    low = resize(arr, (h_small, w_small, arr.shape[2]), anti_aliasing=True, preserve_range=True)
    up = resize(low, (H, W, arr.shape[2]), anti_aliasing=True, preserve_range=True)

    # Cast back to original dtype / range
    if np.issubdtype(orig_dtype, np.integer):
        up = np.clip(np.rint(up), 0, np.iinfo(orig_dtype).max).astype(orig_dtype)
    else:
        up = up.astype(orig_dtype)

    _add_layer_from_hwC(img, up, out_layer)
    return out_layer


def blur_and_boost_contrast(img_arr: np.ndarray, sigma: float = 5, stretch: bool = True) -> np.ndarray:
    """
    Blur an image using a Gaussian filter and (optionally) rescale intensity.

    Parameters
    ----------
    img_arr
        NumPy image array. Expected shape is (H, W, C) or (H, W). If multichannel,
        channels are assumed to be the last axis.
    sigma
        Gaussian sigma to apply (passed to skimage.filters.gaussian).
    stretch
        If True, rescale intensity to full dtype range using skimage.exposure.rescale_intensity.

    Returns
    -------
    np.ndarray
        Blurred image cast to the same dtype as the input.
    """
    blurred = gaussian(img_arr, sigma=sigma, channel_axis=-1 if img_arr.ndim == 3 else None, preserve_range=True)
    if stretch:
        blurred = rescale_intensity(blurred, in_range="image", out_range="dtype")
    return blurred.astype(img_arr.dtype)


def extract_arr_to_df(
    img: xr.Dataset,
    img_layer: str = "image",
    seg_layer: str = "segmented_watershed",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract image arrays, label mask, and region properties as a pandas DataFrame.

    Returns a tuple (arr, labels, df) where:
      - arr is a NumPy array in (C, H, W) layout (channels-first)
      - labels is a 2D integer mask (H, W)
      - df is a pandas.DataFrame combining geometry and per-channel intensity stats

    Requirements / assumptions
    --------------------------
    - The segmentation layer must be 2-D (H, W). If it has library_id/scale dims,
      they are aligned to the first library/scale of the image layer when possible.
    - Region properties are computed using skimage.regionprops_table.

    Parameters
    ----------
    img
        Image container dataset with both image and seg layers.
    img_layer
        Name of the image layer to use for intensity calculations.
    seg_layer
        Name of the segmentation label layer.

    Returns
    -------
    (arr, labels, df)
        arr: np.ndarray (C, H, W)
        labels: np.ndarray (H, W)
        df: pandas.DataFrame
    """
    arr = get_image_CHW(img, layer=img_layer)

    if seg_layer not in img:
        raise KeyError(f"Segmentation layer '{seg_layer}' not found in provided dataset")

    labels = img[seg_layer]

    # Align library_id/scale selection if present on both image and labels
    img_da = img[img_layer]
    if "library_id" in labels.dims and "library_id" in img_da.dims:
        labels = labels.sel(library_id=img_da.coords["library_id"].values[0])
    if "scale" in labels.dims and "scale" in img_da.dims:
        labels = labels.sel(scale=img_da.coords["scale"].values.min())

    labels = labels.data.compute() if hasattr(labels.data, "compute") else labels.to_numpy()
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 2:
        raise ValueError(f"Expected labels to be 2-D (H, W); got {labels.shape}")

    # geometric properties
    geom_props = [
        "label", "area", "perimeter", "eccentricity", "solidity",
        "bbox_area", "major_axis_length", "minor_axis_length",
        "orientation", "centroid"
    ]
    geom_df = pd.DataFrame(regionprops_table(labels, properties=geom_props))

    # per-channel intensity statistics
    dfs = [geom_df]
    for c in range(arr.shape[0]):
        props = ["mean_intensity", "min_intensity", "max_intensity"]
        t = regionprops_table(labels, intensity_image=arr[c], properties=["label"] + props)
        ch_df = (
            pd.DataFrame(t)
            .drop(columns=["label"])
            .add_suffix(f"_ch{c}")
        )
        dfs.append(ch_df)

    df = pd.concat(dfs, axis=1)
    return arr, labels, df


def plot_segmentation_overlay(
    arr: np.ndarray,
    labels: np.ndarray,
    channel_to_show: int = 0,
    use_color: bool = False,
    show_labels: bool = True,
    contour_color: str = "red",
    label_color: str = "yellow",
    figsize: Tuple[float, float] = (10, 10),
    max_labels: Optional[int] = None,
) -> None:
    """
    Overlay segmentation contours and optional numeric labels on an image and show.

    Parameters
    ----------
    arr
        Image array, either (C, H, W) or (H, W). If (C, H, W) and use_color=False,
        `channel_to_show` selects which channel to display. If use_color=True, the
        array is expected to contain 3 or 4 channels (RGB/RGBA) when moved to channels-last.
    labels
        2D segmentation label array (H, W).
    channel_to_show
        Channel index to render when not using color.
    use_color
        If True, display full color image (channels moved to last axis).
    show_labels
        If True, draw integer region labels at centroids.
    contour_color, label_color
        Colors for contour and text respectively.
    figsize
        Figure size tuple.
    max_labels
        If set, limit label text rendering to the first N objects for clarity.
    """
    # Prepare base image
    if arr.ndim == 3:
        if use_color:
            img_disp = np.moveaxis(arr, 0, -1)
        else:
            img_disp = arr[channel_to_show]
    elif arr.ndim == 2:
        img_disp = arr
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}")

    fig, ax = plt.subplots(figsize=figsize)

    # display base image
    if use_color and img_disp.ndim == 3 and img_disp.shape[-1] in (3, 4):
        ax.imshow(np.clip(img_disp, 0, 1))
    else:
        ax.imshow(img_disp, cmap="gray")

    # overlay contours
    ax.contour(labels, colors=contour_color, linewidths=0.6)

    # overlay label IDs
    if show_labels:
        props = regionprops(labels)
        if max_labels is not None:
            props = props[:max_labels]
        for prop in props:
            y, x = prop.centroid
            ax.text(
                x, y, str(prop.label),
                color=label_color,
                fontsize=8,
                ha="center", va="center",
            )

    ax.axis("off")
    plt.tight_layout()
    plt.show()
  
