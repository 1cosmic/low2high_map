# Utils for meta-use in other modules.
import os
import glob
from osgeo import gdal


# Constants
GEO_DATA = 1

# Get the directory of the current module
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(path):
    return os.path.join(MODULE_DIR, path)

DEFAULT_PATH = {
    'images': rel_path('../data/input/images_10m/Sentinel_Samara'),
    'labels': rel_path('../data/input/labels_230m'),
    'processing': rel_path('../data/processing/'),
    'cropped_labels': rel_path('../data/processing/cropped_labels/'),
    'resized_images': rel_path('../data/processing/resized/images/'),
    'resized_labels': rel_path('../data/processing/resized/labels/'),
    'output': rel_path('../data/output/')
}


def check_output():
    pass

def create_output():
    pass


def init():
    print("Initialization paths...")
    for path in DEFAULT_PATH.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
    else:
        print("All paths was initialized.")


def cut_tif_by(src, by, out, mode='mode', resize=False):

    size_x = by['size_x']
    size_y = by['size_y']
    gt = by['transform']
    bounds = [gt[0], gt[3] + gt[5] * size_y,      # minX, minY
            gt[0] + gt[1] * size_x, gt[3]]      # maxX, maxY

    gdal.Warp(
        out,
        src,
        outputBounds=bounds,
        dstSRS=by['projection'],
        xRes=gt[1] if resize else None,
        yRes=gt[5] if resize else None,
        resampleAlg=mode if resize else None,
        dstNodata=0)
    print(f"Map cutted and saved to {out}")


def load_data_tif(data: str,  load_first=False):
    # Load data and geoinfo from the specified path
    if not os.path.exists(data):
        raise ValueError(f"Path not found: {data}")

    results = []
    if os.path.isfile(data):
        data_files = [data]
    else:
        data_files = glob.glob(os.path.join(data, '*.tif'))
        if load_first:
            data_files = data_files[:1]
        if not data_files:
            raise ValueError(f"No data files found in path: {data}")

    for tif in data_files:
        print(f"Loading file: {tif}")
        src = gdal.Open(tif)
        results.append({
            "path": tif,
            "array": src.ReadAsArray(),
            "size_x": src.RasterXSize,
            "size_y": src.RasterYSize,
            "transform": src.GetGeoTransform(),
            "projection": src.GetProjection(),
        })
        src = None  # clear memory
    print("Files was loaded.")

    if load_first:
        return results[0]
    else:
        return results


def save_data_tif(data: dict, path: str, color_palette=None):
    # Save data to the specified path
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    src = data['array']
    if len(src.shape) == 2:
        bands = 1
    elif len(src.shape) == 3:
        bands = src.shape
    else:
        raise ValueError("Data must be 2D or 3D numpy array")

    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(path, data['size_x'], data['size_y'], bands)
    out.SetGeoTransform(data['transform'])
    out.SetProjection(data['projection'])

    if bands == 1:
        out.GetRasterBand(1).WriteArray(src)
    else:
        for i in range(bands):
            out.GetRasterBand(i + 1).WriteArray(src[i])

    if color_palette is not None:
       # Colorpallet
        colors = gdal.ColorTable()
        for class_val, rgb in color_palette.items():
            colors.SetColorEntry(class_val, rgb)
        out.GetRasterBand(1).SetRasterColorTable(colors)
        out.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex) 

    out.FlushCache()
    print(f"Data saved to {path}")