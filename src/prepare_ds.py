# For preparation of dataset

import glob
import os
from utils import init, load_tif, cut_tif_by, save_tif, DEFAULT_PATH, GEO_DATA
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
import itertools
from tqdm import tqdm

init()


def load_resized_data_labels(signs, labels, mask=np.array([]), mode='random', force=False, resize='by_label', percent=0.1, verbose=True):
    if verbose:
        print("Loading & preparing image data...")

    if verbose:
        print("\nCropping & loading labels by 1st image:")
    ref_image = load_tif(signs[0], only_first=True, verbose=verbose)
    cropped_labels = []
    for label in labels:
        name = os.path.basename(label)
        out = DEFAULT_PATH['cropped_labels'] + name
        if not os.path.exists(out) or force:
            cut_tif_by(label, ref_image, out, verbose=verbose)
        else:
            if verbose:
                print("Loading from cache...")
        cropped_labels.append(load_tif(out, only_first=True, verbose=verbose))
    labels = cropped_labels
    if verbose:
        print("Crop labels is done.")

    # Resizing by: label/image.
    if resize == 'by_label':
        if verbose:
            print("\nResizing images by 1st label:")
        sources = signs
        by = labels[0]
        resize_path = 'resized_images'

    elif resize == 'by_sign':
        if verbose:
            print("\nResizing labels by 1st image:")
        sources = labels
        by = load_tif(signs[0], only_first=True, verbose=verbose)
        resize_path = 'resized_labels'
    else:
        raise ValueError("resize can be only [by_sign, by_label]")

    resized = []
    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + name
        if not os.path.exists(out) or force:
            cut_tif_by(src, by, out, resize=True, verbose=verbose)
        else:
            if verbose:
                print("Loading from cache...")
        resized.append(load_tif(out, only_first=True, verbose=verbose))
    if verbose:
        print(f"Resizing {resize} is done.")

    if verbose:
        print("\nData and labels prepared.\n")
    if resize == 'by_label':
        return resized, labels
    elif resize == 'by_sign':
        return signs, resized
    else:
        return [], []


def create_mask(label: dict, image: dict, percent: float = 0.01, r=1,
                select_mode:str = 'random', stratify:bool = False, verbose=True) -> np.ndarray:

    if verbose:
        print("Creating mask...")
    image_array = image['array']
    label_array = label['array']
    filled = image_array > 0
    label_array[~filled] = 0                        # work with ONLY FILLED pixels
    mask = np.zeros_like(label_array, dtype=bool)  # create a mask from the labels
    count_signs = int(filled.size * percent)

    if select_mode == 'random':
        rows, cols = np.where(label_array)
        i = np.random.choice(rows.size, size=count_signs, replace=False)
        i = (rows[i], cols[i])
        mask[i] = True

    elif select_mode == 'secure':
        min_in_neighborhood = minimum_filter(label_array, size=(r*2 + 1))
        max_in_neighborhood = maximum_filter(label_array, size=(r*2 + 1))
        mask_secure_px = (min_in_neighborhood == max_in_neighborhood) & (label_array > 0)
        idx = np.where(mask_secure_px)
        i = np.random.choice(idx[0].size, size=count_signs, replace=False)
        i = (idx[0][i], idx[1][i])
        mask[i] = True
    else:
        raise ValueError(f"Unknown mode: {select_mode}")

    # OPTIMISE THIS FUNC! 
    if stratify:
        classes, count_cl = np.unique(label_array[mask], return_counts=True)
        min_count = count_cl[2]
        balanced = np.zeros_like(label_array, dtype=bool)
        for cl in classes:
            c = min_count if count_cl[cl -1] > min_count else count_cl[cl -1]
            rows, cols = np.where((label_array == cl) & mask)
            i = np.random.choice(rows.size, size=c, replace=False)
            i = (rows[i], cols[i])
            balanced[i] = True
        mask[~balanced] = False

    output = DEFAULT_PATH['output']
    name = f'mask_{select_mode}_r{r}_{percent}'
    if stratify:
        name = name + '_stratified.tif'
    else:
        name = name + '.tif'

    out = label.copy()
    out['array'] = mask
    save_tif(out, output + name, with_bg=True, verbose=verbose)
    if verbose:
        print("Mask created.\n")

    return mask


def stack_and_zip(signs: list, labels: list, mask: np.ndarray, verbose=True):

    if verbose:
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

    zip_signs = zip_signs[mask]
    zip_labels = zip_labels[mask]

    if verbose:
        print(f"Size dataset before -> after zip by mask:")
        print("signs:", signs[0]['array'].size * len(signs), '->', zip_signs.size, '| bands:', len(tensor.keys()), '| shape:', zip_signs.shape)
        print("labels:", labels[0]['array'].size * len(labels), '->', zip_labels.size, '| shape:', zip_labels.shape)
    
    return zip_signs, zip_labels


# Iteration of dataset for training with transform data to tensor.
def generate_dataset(signs, labels, percent, force=False, mask_mode='random', r=1, stratify=False, resize='by_label', verbose=True):
    signs, labels = load_resized_data_labels(signs, labels, resize=resize, force=force, verbose=verbose)
    mask = create_mask(labels[0], signs[0], percent=percent, select_mode=mask_mode, r=r, stratify=stratify, verbose=verbose)
    zip_signs, zip_labels = stack_and_zip(signs, labels, mask, verbose=verbose)

    return zip_signs, zip_labels, signs, labels