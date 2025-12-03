# For preparation of dataset

from time import time
import gc
import os
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, sobel
from skimage.transform import resize

from utils import init, load_tif, cut_tif_by, save_tif, DEFAULT_PATH, GEO_DATA

import itertools
from tqdm import tqdm

init()


def load_resized_data_labels(signs, labels, force=False, resize='by_label', verbose=True):
    if verbose:
        print("Loading & preparing image data...")

    if verbose:
        print("\nCropping & loading labels by 1st image:")
    ref_image = load_tif(signs[0], only_first=True, verbose=verbose)
    cropped_labels = []
    for label in labels:
        print(label)
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


def create_homogeneous_layer(signs, out=None, resize_by=None, force=False, verbose=False):
    r_path = None
    nir_path = None
    for path in signs:
        if '.r.' in path:
            r_path = path
        elif '.n.' in path:
            nir_path = path
    if r_path is None or nir_path is None:
        Exception("Please, check what in sings_path are exists: r & nir tiles.")

    if out is None:
        out = DEFAULT_PATH['processing'] + 'homogeneous_layer.tif'

    res = None
    if os.path.exists(out) and not force:
        print("Finded gomogeneous layer. Load it.")
        res = load_tif(out, only_first=True, verbose=verbose)

    else:
        r_obj = load_tif(r_path, only_first=True, verbose=verbose)
        nir_obj = load_tif(nir_path, only_first=True, verbose=verbose)
        print("Retyping...")
        r = r_obj['array'].astype(np.float32)
        nir = nir_obj['array'].astype(np.float32)
        del r_obj['array'], nir_obj
        gc.collect()

        Mpx = 1000000
        chunk = 10 * 1024^2
        N = r.size / Mpx  # r and nir have same shape
        size = r.shape

        print("NDVI…")
        t0 = time()
        homogeneous_array = (r - nir) / (r + nir + 1e-6)
        dt = time() - t0
        print(f"NDVI done: {dt:.3f}s   | per Mpx: {dt/N:.3f}s")

        t1 = time()
        gx = sobel(homogeneous_array, axis=1)
        dt = time() - t1
        print(f"gx done: {dt:.3f}s     | per Mpx: {dt/N:.3f}s")

        t2 = time()
        gy = sobel(homogeneous_array, axis=0)
        dt = time() - t2
        print(f"gy done: {dt:.3f}s     | per Mpx: {dt/N:.3f}s")

        t3 = time()
        gx = gx.ravel(); gy = gy.ravel()
        homogeneous_array = np.zeros_like(gx)
        for i in tqdm(range(0, gy.size, chunk)):
            homogeneous_array[i:i+chunk] = np.hypot(gx[i:i+chunk], gy[i:i+chunk])
        homogeneous_array = homogeneous_array.reshape(size)
        dt = time() - t3
        del gx, gy
        gc.collect()
        print(f"hypot done: {dt:.3f}s  | per Mpx: {dt/N:.3f}s")

        t4 = time()
        homogeneous_array = uniform_filter(homogeneous_array, size=23)
        dt = time() - t4
        print(f"uniform_filter done: {dt:.3f}s | per Mpx: {dt/N:.3f}s")
        total = time() - t0
        print(f"total: {total:.3f}s | total per Mpx: {total/N:.3f}s")

        r_obj['array'] = homogeneous_array
        save_tif(r_obj, out, save_as_float=True, verbose=verbose)
        res = r_obj


    if resize_by is not None:
        resize_by = load_tif(resize_by, only_first=True)
        print("Start resizing of homoheneous layer.")
        resized_out = DEFAULT_PATH['resized_layers'] + 'homogeneous_layer.tif'
        if not os.path.exists(resized_out) or force:
            cut_tif_by(out, resize_by, resized_out, resize=True, mode='bilinear',verbose=verbose)
        res = load_tif(resized_out, only_first=True)

    return res


def create_mask(label: dict, image: dict, percent: float = 0.01, r=1,
                mode:str = 'random', homogen_layer = None, homogen_percent=0.1, stratify:bool = False, verbose=True) -> np.ndarray:

    if verbose:
        print("Creating mask...")
    image_array = image['array']
    label_array = label['array']
    label_array[image_array <= 0] = 0              # work with ONLY FILLED pixels
    mask = np.zeros_like(label_array, dtype=bool)  # create a mask from the labels

    # TODO: ERROR WITH COMBINE 'HOMOGENEOUS'!
    if stratify:
        count_signs = label_array[label_array > 0].size
    else:
        count_signs = int(label_array[label_array > 0].size * percent)

    if mode == 'random':
        rows, cols = np.where(label_array)
        idx = np.random.choice(rows.size, size=count_signs, replace=False)
        idx = (rows[idx], cols[idx])
        mask[idx] = True

    elif mode == 'secure':
        min_in_neighborhood = minimum_filter(label_array, size=(r*2 + 1))
        max_in_neighborhood = maximum_filter(label_array, size=(r*2 + 1))
        mask_secure_px = (min_in_neighborhood == max_in_neighborhood) & (label_array > 0)
        idx = np.where(mask_secure_px)
        count = count_signs if count_signs <= idx[0].size else idx[0].size
        print("count of value in secure mask: ", count)
        i = np.random.choice(idx[0].size, size=count, replace=False)
        idx = (idx[0][i], idx[1][i])
        mask[idx] = True

    elif mode == 'homogeneous':
        if homogen_layer is None:
            raise Exception("Sorry, you don't provide homogeneous mask.")
        homogen_layer = homogen_layer['array']
        if homogen_layer.shape != label_array.shape:
            raise Exception(f"Homogeneous shape: {homogen_layer.shape} != label: {label_array.shape}")

        homogen_mask = label_array > 0
        mean_homogeneous = np.bincount(label_array[homogen_mask], weights=homogen_layer[homogen_mask])[1:]
        sums = np.bincount(label_array[homogen_mask])[1:]
        # print("accum_brights:", mean_homogeneous)
        mean_homogeneous = mean_homogeneous / sums
        # print("sums:", sums)
        # print("means: ", mean_homogeneous)

        homogen_mask = np.zeros_like(label_array, dtype=bool)
        for cl in range(len(mean_homogeneous)):
            mean_class = mean_homogeneous[cl]
            min_mean = mean_class - homogen_percent / 2
            max_mean = mean_class + homogen_percent / 2
            
            cursor = ((homogen_layer >= min_mean) &
                      (homogen_layer <= max_mean) & 
                      (label_array == cl + 1))
            homogen_mask[cursor] = 1
            # print(f"for cl: {cl+1} true px: ", cursor[cursor].sum())

        rows, cols = np.where(homogen_mask)
        count_signs = int(rows.size * percent) if not stratify else rows.size
        i = np.random.choice(rows.size, size=count_signs, replace=False)
        idx = (rows[i], cols[i])
        mask[idx] = 1

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # OPTIMISE THIS FUNC! 
    if stratify:
        classes, counts = np.unique(label_array[mask], return_counts=True)
        print("Stratify output:", classes, counts)
        balanced = np.zeros_like(label_array, dtype=bool)
        min_c = counts.min()

        for c in classes:
            rows, cols = np.where((label_array == c) & mask)
            idx = np.random.choice(rows.size, size=min_c, replace=False)
            idx = (rows[idx], cols[idx])
            balanced[idx] = True
        mask[~balanced] = False

    print(f"Mode: {mode} & stratify {stratify}:", np.unique(label_array[mask], return_counts=True))

    output = DEFAULT_PATH['output']
    name = f'mask_{mode}_r{r}_{percent}'
    if mode == "homogeneous":
        name = name + f'hp_{homogen_percent}'
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
def generate_dataset(signs, labels, percent, force=False, mask_mode='random', r=1, homogen_percent=0.01, stratify=False, resize='by_label', verbose=True):


    signs, labels = load_resized_data_labels(signs, labels, resize=resize, force=force, verbose=verbose)

    if mask_mode == 'homogeneous':
        resize_by = labels[0]['path'] if resize == 'by_label' else None
        homogeneous = create_homogeneous_layer(signs, out=None, resize_by=resize_by, verbose=verbose)
    else:
        homogeneous = None

    mask = create_mask(labels[0], signs[0], mode=mask_mode, 
                       percent=percent, stratify=stratify, 
                       r=r, homogen_layer = homogeneous, homogen_percent=homogen_percent, verbose=verbose)
    zip_signs, zip_labels = stack_and_zip(signs, labels, mask, verbose=verbose)

    return zip_signs, zip_labels, signs, labels