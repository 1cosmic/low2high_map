from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import glob
import os

from utils import DEFAULT_PATH, cut_tif_by, load_data_tif


def validate_how_array(predicted, etalon):
    if predicted.shape != etalon.shape:
        raise Exception(f"Shape of 'predicted' {predicted.shape} != {etalon.shape} 'etalon'.")
    
    return {'classification_report': classification_report(predicted, etalon),
            'confusion_matrix': confusion_matrix(predicted, etalon)}


def cut_etalons_in_path(predicted, etalons=None):
    etalons = DEFAULT_PATH['etalons'] if etalons is None else etalons
    etalons = glob.glob(os.path.join(etalons, '*.tif'))

    for etalon in etalons:
        name = os.path.basename(etalon)
        out = DEFAULT_PATH['resized_etalons'] + name
        cut_tif_by(predicted, etalon, out, mode='mode')

    return glob.glob(os.path.join(DEFAULT_PATH['resized_etalons']))


def validate_how_tif(predicted_path:str, etalons_path:list):
    etalons = cut_etalons_in_path(predicted_path, etalons_path)
    loaded_predicted = load_data_tif(predicted_path)

    for etalon in etalons:
        loaded_etalon = load_data_tif(etalon)
        res = validate_how_array(loaded_predicted, loaded_etalon)
        print(res)