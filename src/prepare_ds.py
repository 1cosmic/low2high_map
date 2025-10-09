# For preparation of dataset

from utils import init, load_data_tif, cut_map_by, save_data_tif, DEFAULT_PATH, GEO_DATA
import visualisation as vis

import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt

init()


def create_mask(label: np.ndarray, data: np.ndarray = np.array([]),
                mode: str = 'random', percent: float = 0.1) -> np.ndarray:
    # Create a mask from the labels
    mask = np.zeros_like(label)

    if mode == 'random':
        indices = np.random.choice(np.arange(label.size),
                                   size=int(label.size * percent), replace=False)
        mask.ravel()[indices] = 1

    elif mode == 'not_border':
        pass

    elif mode == 'stratified':
        pass

    elif mode == 'balanced':
        pass

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mask


def preparation_data(images=DEFAULT_PATH['images'], labels=DEFAULT_PATH['labels'], mask=np.array([]), mode='random', resize='by_label', percent=0.1):

    images = load_data_tif(images)
    labels = load_data_tif(labels)

    if resize == 'by_label':
        print("Resizing images by label...")
        for i in range(len(images)):
            image = DEFAULT_PATH['processing'] + f'image_by_label_{i}.tif'
            cut_map_by(images[i][0], labels[0][GEO_DATA], image, resize=True)
            images[i] = load_data_tif(image, return_first=True)
        print("Images resized by label.\n")

    elif resize == 'by_image':
        print("Resizing labels by image...")
        for i in range(len(labels)):
            label = DEFAULT_PATH['processing'] + f'label_by_image{i}.tif'
            cut_map_by(labels[i][0], images[0][GEO_DATA], label, resize=True)
            labels[i] = load_data_tif(label, return_first=True)
        print("Labels resized by image.\n")

    data = None
    if mode == 'random':
        pass
        

    elif mode == 'stratified':
        pass

    elif mode == 'balanced':
        pass

    return data


def preparation_dataset(mode='random', resize='by_label', percent=0.01):
    # Preparation of dataset for training, transform for numpy.

    # Prepare of label: cut it by photo.
    print("Cutting label by image...")
    label = load_data_tif(DEFAULT_PATH['labels'])
    image = load_data_tif(DEFAULT_PATH['images'], load_first=True)
    cutted_map = DEFAULT_PATH['output'] + 'cutted_map.tif'
    cut_map_by(label[0][0], image[0][1], cutted_map)
    print("Label cutted.\n")

    # Prepare labels and data
    print("Preparing data and labels...")
    label = load_data_tif(cutted_map, return_first=True)
    mask = create_mask(label[GEO_DATA]['array'], mode=mode, percent=percent)
    data = preparation_data(images=DEFAULT_PATH['images'], labels=DEFAULT_PATH['labels'], mask=mask, mode=mode, percent=percent)
    print("Data and labels prepared.\n")

    # TODO: save processed data processed/tmp folder
    # ...

    # TODO: process data to numpy arrays for training
    # ...

    return None


# TODO: extract bottom to test module
def test_create_mask(percent=0.01):
    # Test function

    label = load_data_tif(DEFAULT_PATH['labels'])[0]
    data = label[GEO_DATA]
    mask = create_mask(data['array'], mode='random', percent=percent)
    out = data.copy()
    out['array'] = mask

    save_data_tif(out, f"{DEFAULT_PATH['output']}/mask_{percent}.tif")
    vis.plot_data(data['array'], mask=mask)
# test_create_mask()


def preparation_data_test():
    # Test function for preparation of data
    res = preparation_data(data_path=DEFAULT_PATH['images'],
                           data_labels=DEFAULT_PATH['labels'], mode='random', percent=0.01)
    if res is not None:
        for item in res:
            print(item[0], item[1]['array'].shape, item[1]['size_x'], item[1]['size_y'])
            vis.plot_data(item[1]['array'])
    else:
        print("No data loaded.")
# preparation_data_test()