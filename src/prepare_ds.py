# For preparation of dataset

import glob
import os
from utils import init, load_data_tif, cut_tif_by, save_data_tif, DEFAULT_PATH, GEO_DATA
import visualisation as vis
import numpy as np

init()


def create_mask(label: dict, image: dict,
                mode: str = 'random', percent: float = 0.01, output: str = '') -> np.ndarray:

    label_array = label['array']
    image_array = image['array']
    
    # Create a mask from the labels
    mask = np.zeros_like(label_array, dtype=bool)
    rows, cols = np.where(image_array > 0)

    # Get random indices ONLY FILLED pixels.
    if mode == 'random':
        i = np.random.choice(rows.size, int(percent * rows.size), replace=False)
        i_pixel = (rows[i], cols[i])
        mask[i_pixel] = True

    elif mode == 'not_border':
        pass

    elif mode == 'stratified':
        pass

    elif mode == 'balanced':
        pass

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if output:
        out = label.copy()
        out['array'] = mask
        save_data_tif(out, output)

    return mask


def load_resized_data_labels(images=DEFAULT_PATH['images'], labels=DEFAULT_PATH['labels'], mask=np.array([]), mode='random', resize='by_label', percent=0.1):

    images = glob.glob(os.path.join(images, '*.tif'))
    labels = glob.glob(os.path.join(labels, '*.tif'))

    print("\nCropping labels by 1st image:")
    for label in labels:
        name = os.path.basename(label)
        out = DEFAULT_PATH['cropped_labels'] + name
        cut_tif_by(label, load_data_tif(images[0], load_first=True), out)
    print("Cropping done. Loading in memory...")
    labels = load_data_tif(DEFAULT_PATH['cropped_labels'])

    if resize == 'by_label':
        print("\nResizing images by 1st label:")
        sources = images
        by = labels[0]
        resize_path = 'resized_images'

    elif resize == 'by_image':
        print("\nResizing labels by 1st image:")
        sources = labels
        by = load_data_tif(images[0], load_first=True)
        resize_path = 'resized_labels'

    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + name
        cut_tif_by(src, by, out, resize=True)

    print("Resizing done. Loading in memory...")
    images = load_data_tif(DEFAULT_PATH['resized_images'])

    print(images)
    print(labels)
    return images, labels


def preparation_dataset(mask_mode='random', resize='by_label', percent=0.01):
    # Preparation of dataset for training, transform for numpy.

    print("Loading & preparing image data...")
    data, labels = load_resized_data_labels(resize=resize)
    print("\nData and labels prepared.\n")

    print("Creating mask...")
    mask_tif = DEFAULT_PATH['processing'] + 'mask_only_filled.tif'
    image = data[0]
    label = labels[0]
    mask = create_mask(label, image, mode=mask_mode, percent=percent, output=mask_tif)
    print("Mask created.\n")


    # TODO: save processed data processed/tmp folder
    # ...

    # TODO: process data to numpy arrays for training
    # ...

    return data, labels


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