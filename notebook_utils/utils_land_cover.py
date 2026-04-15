"""Scripts for land cover data processing and visualization."""


import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.windows import from_bounds
from owslib.wms import WebMapService
from rasterio.io import MemoryFile
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def open_image(img_path):
    """Open a raster image using rioxarray and print its metadata."""
    img = rioxarray.open_rasterio(img_path)
    print(f"Image shape: {img.shape}")
    print(f"Bands: {img.band.values}")
    print(f"CRS: {img.rio.crs}")
    return img


def clean_image(img):
    img_clean = img.copy().astype("float32") 
    if 'long_name' in img.attrs: 
        list_name_band = img.attrs['long_name']
        print(f"Bands list: {list_name_band}")

    bands = img.band.values
    

    all_values = img_clean.values[~np.isnan(img.values)]
    negatives_values = all_values[all_values < 0]
    print(f"  negative values found: {negatives_values}")

    img_clean = img_clean.where(img_clean > 0) # Replace Negative values with Nan
    scl = img_clean.sel(band=bands[-1])
    print("SCL max: ",scl.values.max())
    if scl.max() < 1:
        scl = (scl *10000).astype("int") # All the bands were divided by 10,000 during downloading, so I have to scale them back to a class between 0 and 11

    valid_pixels = scl.isin([4, 5, 6])  # Vegetation, Not-vegetated, Water
    img_clean = img_clean.where(valid_pixels)

    for b in range(0, len(bands)-1):
        band = bands[b] 
        band_data = img_clean.sel(band=band)
        
        # Si valeurs en DN (> 1), convertir en réflectance
        if float(band_data.max()) > 20.0:
            band_data = band_data / 10000.0
 
        # Median Absolute Deviation
 
        valid = band_data.values[~np.isnan(band_data.values)]
        if len(valid) > 100:
            q1, q3 = np.percentile(valid, [25, 75])
            iqr = q3 - q1
            band_data = band_data.where((band_data >= q1 - 5*iqr) & (band_data <= q3 + 5*iqr))
        img_clean.loc[dict(band=band)] = band_data


    return img_clean


def display_rgb(img, bands=[3,2,1], ax=None, index_figure=[0,0], title="RGB Image"):
    """Display an RGB image from Sentinel-2 data."""
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        ax = np.atleast_2d(ax)
        index_figure = [0,0]
    else:
        ax = np.atleast_2d(ax)
    img.sel(band=bands).plot.imshow(ax=ax[index_figure[0], index_figure[1]], robust=True)
    ax[index_figure[0], index_figure[1]].axis('off')
    ax[index_figure[0], index_figure[1]].set_title(title)
    ax[index_figure[0], index_figure[1]].set_xlabel("Easting (m)")
    ax[index_figure[0], index_figure[1]].set_ylabel("Northing (m)")


def compute_image_statistics(img):
    
    list_name_band = img.attrs['long_name']
    if isinstance(list_name_band, str):
        list_name_band = [name.strip() for name in list_name_band.split(",")]  # Convert string to list if necessary
    bands = img.band.values


    print("=" * 90)
    print(f"{'':<35}DATA STATISTICS")
    print("=" * 90)
    print(f"\n{'Band':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'98th percentile':<18} {'Nan values (%)':<10}")
    print("-" * 90)

    for b in range(0, len(bands)-1):
        band = bands[b]
        band_da = img.sel(band=band)               
        data_array = band_da.values.flatten()
    
        min_val = float(np.nanmin(band_da))
        max_val = float(np.nanmax(band_da))
        mean_val = float(np.nanmean(band_da))
        std_val = float(np.nanstd(band_da))
        p98 = float(band_da.quantile(0.98, skipna=True))

        # Count NaN values
        nan_count = np.isnan(data_array).sum()
        nan_percent = (nan_count / len(data_array)) * 100
        
    
        print(f"{list_name_band[b]:<10} {min_val:<10.3f} {max_val:<10.3f} {mean_val:<10.3f} "
            f"{std_val:<15.3f} {p98:<17.3f} {nan_percent:<10.2f}")

    
