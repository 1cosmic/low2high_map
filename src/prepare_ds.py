# For preparation of dataset

import glob
import os
from utils import init, load_data_tif, cut_tif_by, save_data_tif, DEFAULT_PATH, GEO_DATA
import visualisation as vis
import numpy as np

init()


def create_mask(label: dict, image: dict,
                mode: str = 'random', percent: float = 0.01, output: str = '') -> np.ndarray:

    print("Creating mask...")

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
    print("Mask created.\n")

    return mask


def load_resized_data_labels(data=DEFAULT_PATH['images'], labels=DEFAULT_PATH['labels'], mask=np.array([]), mode='random', resize='by_label', percent=0.1):

    print("Loading & preparing image data...")
    data = glob.glob(os.path.join(data, '*.tif'))
    labels = glob.glob(os.path.join(labels, '*.tif'))

    print("\nCropping labels by 1st image:")
    for label in labels:
        name = os.path.basename(label)
        out = DEFAULT_PATH['cropped_labels'] + name
        cut_tif_by(label, load_data_tif(data[0], load_first=True), out)
    print("Cropping done. Loading in memory...")
    labels = load_data_tif(DEFAULT_PATH['cropped_labels'])

    if resize == 'by_label':
        print("\nResizing images by 1st label:")
        sources = data
        by = labels[0]
        resize_path = 'resized_images'

    elif resize == 'by_image':
        print("\nResizing labels by 1st image:")
        sources = labels
        by = load_data_tif(data[0], load_first=True)
        resize_path = 'resized_labels'

    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + name
        cut_tif_by(src, by, out, resize=True)

    print("Resizing done. Loading in memory...")
    data = load_data_tif(DEFAULT_PATH['resized_images'])
    print("\nData and labels prepared.\n")

    return data, labels


def zip_dataset(data: dict, labels: dict, mask: np.ndarray):

    print("Zipping dataset by mask...")
    zip_data = [d['array'][mask > 0] for d in data]
    zip_labels = [l['array'][mask > 0] for l in labels]

    print("create tensor by bands:")
    tensor = {}
    for d in data:
        band_name = d['path'].replace('..', '').split('.')[1]
        # print(band_name)
        layer = d['array'][mask > 0]
        if band_name in tensor.keys():
            tensor[band_name] = np.dstack((tensor[band_name], layer))
        else:
            tensor[band_name] = layer

    # zip_data = np.array([tensor[band] for band in tensor.keys()])  # shape (4 x n)
    zip_data = np.dstack([tensor[band] for band in tensor.keys()])   # shape (n x 4)
    zip_data = np.squeeze(zip_data)     # TODO: remove it after adding of layer-year
    # TODO: do shape (pixel x band x year)!

    zip_labels = np.squeeze(np.array(zip_labels))

    print("Size dataset before -> after zip:")
    print("data:", data[0]['array'].size * len(data), '->', zip_data.size, '| bands:', len(tensor.keys()), '| shape:', zip_data.shape)
    print("labels:", labels[0]['array'].size * len(labels), '->', zip_labels.size, '| shape:', zip_labels.shape)
    
    return zip_data, zip_labels


def preparation_dataset(mask_mode='random', resize='by_label', percent=0.01):
    # Preparation of dataset for training, transform for numpy.

    data, labels = load_resized_data_labels(resize=resize)
    out_mask = DEFAULT_PATH['output'] + f'mask_only_filled_{mask_mode}_{percent}.tif'
    mask = create_mask(labels[0], data[0], mode=mask_mode, percent=percent, output=out_mask)
    zip_data, zip_labels = zip_dataset(data, labels, mask)

    # TODO: process data to numpy arrays for training
    # ...

    return zip_data, zip_labels


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