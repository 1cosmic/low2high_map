# For preparation of dataset

from utils import init, load_data_tif, cut_map_by, save_data_tif, DEFAULT_PATH, GEO_DATA
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


def load_resized_data_labels(data=DEFAULT_PATH['images'], labels=DEFAULT_PATH['labels'], mask=np.array([]), mode='random', resize='by_label', percent=0.1):

    data = load_data_tif(data)
    labels = load_data_tif(labels)

    # Resize for test: low/high resolution of map.
    print()
    if resize == 'by_label':
        print("Resizing images by label...")
        for i in range(len(data)):
            image = DEFAULT_PATH['processing'] + f'image_by_label_{i}.tif'
            cut_map_by(data[i][0], labels[0][GEO_DATA], image, resize=True)
            data[i] = load_data_tif(image, return_first=True)
        print("Images resized by label.\n")

    elif resize == 'by_image':
        print("Resizing labels by image...")
        for i in range(len(labels)):
            label = DEFAULT_PATH['processing'] + f'label_by_image{i}.tif'
            cut_map_by(labels[i][0], data[0][GEO_DATA], label, resize=True)
            labels[i] = load_data_tif(label, return_first=True)
        print("Labels resized by image.\n")

    # if mode == 'random':
    #     pass

    # elif mode == 'stratified':
    #     pass

    # elif mode == 'balanced':
    #     pass

    return data, labels


def preparation_dataset(mask_mode='random', resize='by_label', percent=0.01):
    # Preparation of dataset for training, transform for numpy.

    print("Loading & preparing image data...")
    data, labels = load_resized_data_labels(resize=resize)
    print("Data and labels prepared.\n")

    print("Creating mask...")
    mask_tif = DEFAULT_PATH['processing'] + 'mask_only_filled.tif'
    image = data[0][GEO_DATA]
    label = labels[0][GEO_DATA]
    mask = create_mask(label, image, mode=mask_mode, percent=percent, output=mask_tif)
    print("Mask created.\n")


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