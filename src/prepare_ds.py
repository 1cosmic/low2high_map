# For preparation of dataset

import glob
import os
from utils import init, load_data_tif, cut_tif_by, save_data_tif, DEFAULT_PATH, GEO_DATA
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


def load_resized_data_labels(signs, labels, mask=np.array([]), mode='random', force=False, resize='by_label', percent=0.1):
    print("Loading & preparing image data...")

    print("\nCropping & loading labels by 1st image:")
    ref_image = load_data_tif(signs[0], only_first=True)
    cropped_labels = []
    for label in labels:
        name = os.path.basename(label)
        out = DEFAULT_PATH['cropped_labels'] + name
        if not os.path.exists(out) or force:
            cut_tif_by(label, ref_image, out)
        else:
            print("Loading from cache...")
        cropped_labels.append(load_data_tif(out, only_first=True))
    labels = cropped_labels
    print("Crop labels is done.")

    # Downgrade of classes on label-map.
    

    # Resizing by: label/image.
    if resize == 'by_label':
        print("\nResizing images by 1st label:")
        sources = signs
        by = labels[0]
        resize_path = 'resized_images'

    elif resize == 'by_sign':
        print("\nResizing labels by 1st image:")
        sources = labels
        by = load_data_tif(signs[0], only_first=True)
        resize_path = 'resized_labels'
    else:
        raise ValueError("resize can be only [by_sign, by_label]")

    resized = []
    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + name
        if not os.path.exists(out) or force:
            cut_tif_by(src, by, out, resize=True)
        else:
            print("Loading from cache...")
        resized.append(load_data_tif(out, only_first=True))
    print(f"Resizing {resize} is done.")

    print("\nData and labels prepared.\n")
    if resize == 'by_label':
        return resized, labels
    elif resize == 'by_sign':
        return signs, resized


def stack_and_zip(signs: list, labels: list, mask: np.ndarray):

    print("Zipping dataset by mask...")
    print("create tensor by bands")
    tensor = {}
    for d in signs:
        band_name = d['path'].replace('..', '').split('.')[1]
        layer = d['array']
        if band_name in tensor.keys():
            tensor[band_name] = np.dstack((tensor[band_name], layer))
        else:
            tensor[band_name] = layer

    zip_signs = np.dstack([tensor[band] for band in tensor.keys()])   # shape (n x 4)
    zip_signs = np.squeeze(zip_signs)
    zip_labels = [l['array'] for l in labels]
    zip_labels = np.squeeze(np.array(zip_labels))

    zip_signs = zip_signs[mask > 0]
    zip_labels = zip_labels[mask > 0]

    print(f"Size dataset before -> after zip by mask:")
    print("signs:", signs[0]['array'].size * len(signs), '->', zip_signs.size, '| bands:', len(tensor.keys()), '| shape:', zip_signs.shape)
    print("labels:", labels[0]['array'].size * len(labels), '->', zip_labels.size, '| shape:', zip_labels.shape)
    
    return zip_signs, zip_labels


def generate_dataset(signs, labels, mask_mode='random', resize='by_label', percent=0.01):
    # Iteration of dataset for training with transform data to tensor.

    signs, labels = load_resized_data_labels(signs, labels, resize=resize)
    out_mask = DEFAULT_PATH['output'] + f'mask_only_filled_{mask_mode}_{percent}.tif'
    mask = create_mask(labels[0], signs[0], mode=mask_mode, percent=percent, output=out_mask)
    zip_signs, zip_labels = stack_and_zip(signs, labels, mask)

    return zip_signs, zip_labels, signs, labels