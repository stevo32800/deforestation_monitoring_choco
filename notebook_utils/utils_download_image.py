import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds



def read_band_pixels(item, band_name, bbox):
    """Lit les pixels d'une bande Landsat croppée sur la bbox"""
    with rasterio.open(item.assets[band_name].href) as src:
        # Crop the tile to the bbox in the native CRS 
        bounds_native = transform_bounds('EPSG:4326', src.crs, *bbox)
        window = from_bounds(*bounds_native, src.transform)
        data = src.read(1, window=window).astype('float32')
    # Appliquer le facteur d'échelle Landsat Collection 2 ou Sentinel-2
    data = data /10000 if 'sentinel-2' in item.collection_id else data * 0.0000275 - 0.2
    
    return data

def display_image(item, rgb_name, bbox=None):
    """
    Affiche une image RGB
    
    Parameters:
    -----------
    item : pystac.Item
        Item STAC
    rgb_name : list
        Liste des bandes [R, G, B], ex: ['B04', 'B03', 'B02']
    bbox : list, optional
        [west, south, east, north] en WGS84 pour cropper
    """
    
    print(f"🎨 Création d'une composition RGB...")
    red_pixels   = read_band_pixels(item, rgb_name[0], bbox)
    green_pixels = read_band_pixels(item, rgb_name[1], bbox)
    blue_pixels  = read_band_pixels(item, rgb_name[2], bbox)
    print(red_pixels.shape, green_pixels.shape, blue_pixels.shape)

    """    # Masquer les nodata (fill value DN=0 -> valeur négative après scaling)
    nodata_mask = (red_pixels < 0) | (green_pixels < 0) | (blue_pixels < 0)
    red_pixels[nodata_mask]   = np.nan
    green_pixels[nodata_mask] = np.nan
    blue_pixels[nodata_mask]  = np.nan"""

    def stretch(band):
        """Étirement linéaire percentile 2%-98% pour améliorer le contraste"""
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            return np.zeros_like(band)  # ou np.nan * band
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        return np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)

    rgb = np.dstack([stretch(red_pixels), stretch(green_pixels), stretch(blue_pixels)])
    #rgb[nodata_mask] = 0  # nodata en noir

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(rgb)
    ax.set_title(
        f"Composition RGB - {item.datetime.strftime('%Y-%m-%d')}",
        fontsize=16, 
        fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def afficher_rgb(dataset, rgb_bands=['B04', 'B03', 'B02'], date_idx=0):
    red   = dataset[rgb_bands[0]][date_idx, :, :].values.astype('float32')
    green = dataset[rgb_bands[1]][date_idx, :, :].values.astype('float32')
    blue  = dataset[rgb_bands[2]][date_idx, :, :].values.astype('float32')
    satelite_type = 'Sentinel-2' if 'B0' in rgb_bands[0] else 'Landsat'

    if satelite_type == 'Sentinel-2':
        red, green, blue = red / 10000, green / 10000, blue / 10000
    else:
        red   = red   * 0.0000275 - 0.2
        green = green * 0.0000275 - 0.2
        blue  = blue  * 0.0000275 - 0.2

    # Masquer les nodata (fill value DN=0 -> valeur negative apres scaling)
    nodata_mask = (red < 0) | (green < 0) | (blue < 0)
    red[nodata_mask]   = np.nan
    green[nodata_mask] = np.nan
    blue[nodata_mask]  = np.nan

    def stretch(band):
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            return np.zeros_like(band)  # ou np.nan * band
        p2, p98 = np.percentile(valid, 1), np.percentile(valid, 99)
        return np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    rgb[nodata_mask] = 0  # nodata en noir

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"RGB {satelite_type} - {dataset.time.values[date_idx].astype('datetime64[D]')}",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_histogram(item, rgb_name, bbox=None):
    """
    Affiche les histogrammes des bandes RGB et leurs statistiques de base
    Parameters:
    -----------
    item : pystac.Item
        Item STAC
    rgb_name : list
        Liste des bandes [R, G, B], ex: ['B04', 'B03', 'B02']
    bbox : list, optional
        [west, south, east, north] en WGS84 pour cropper
    """

    red_pixels   = read_band_pixels(item, rgb_name[0],   bbox)
    green_pixels = read_band_pixels(item, rgb_name[1], bbox)
    blue_pixels  = read_band_pixels(item, rgb_name[2],  bbox)

    plt.figure(figsize=(12, 5))
    plt.hist(red_pixels.flatten(),   bins=100, color='red',   alpha=0.5, label=f'Red ({rgb_name[0]})',   density=True)
    plt.hist(green_pixels.flatten(), bins=100, color='green', alpha=0.5, label=f'Green ({rgb_name[1]})', density=True)
    plt.hist(blue_pixels.flatten(),  bins=100, color='blue',  alpha=0.5, label=f'Blue ({rgb_name[2]})',  density=True)
    plt.title(f"RGB histogram bands ({item.datetime.strftime('%Y-%m-%d')})", fontsize=14, fontweight='bold')
    plt.xlabel("Surface reflectance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Red   — min: {red_pixels.min():.4f} | max: {red_pixels.max():.4f} | mean: {red_pixels.mean():.4f}")
    print(f"Green — min: {green_pixels.min():.4f} | max: {green_pixels.max():.4f} | mean: {green_pixels.mean():.4f}")
    print(f"Blue  — min: {blue_pixels.min():.4f} | max: {blue_pixels.max():.4f} | mean: {blue_pixels.mean():.4f}")


