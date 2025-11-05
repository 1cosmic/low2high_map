# Module 2: create map by predict on weights
import joblib
import numpy as np
from tqdm import tqdm

from utils import save_data_tif, DEFAULT_PATH

color_palette = {
    0: (255, 0, 0),       # Class 0 (ошибки): FF0000
    1: (98, 111, 71),     # Class 1 (лес): 626F47
    2: (232, 184, 109),   # Class 2 (пахотные): E8B86D
    3: (210, 255, 114),   # Class 3 (луг): D2FF72
    4: (3, 174, 210),     # Class 4 (вода): 03AED2
    5: (179, 200, 207),   # Class 5 (урбанизация): B3C8CF
    6: (253, 250, 246)    # Class 6 (прочие): FDFAF6
}


def create_map(images: dict, model, name: str, count_chunks=2):
    # model = joblib.load(model)
    X = np.dstack([img['array'] for img in images])
    s_x = X.shape[0]
    s_y = X.shape[1]
    band = X.shape[2]
    print("Reshaping tensor-images...")

    # TOOD: X.reshape() for Full reshape's very slow.
    # D.S. say, what X{chunks}.reshape's can be fast.
    X = X.reshape(-1, band)

    print("Start create of map...")
    print(f"\nmap size: x = {s_x}, y = {s_y}, bands = {band}, total px={s_x * s_y}")

    MB = pow(1024, 2)
    chunk_s = int(count_chunks * MB) # adjust based on your RAM
    if len(X) < chunk_s:
        chunk_s = int(len(X) / 5)

    predict_map = np.zeros(X.shape[0], dtype=np.uint8)
    for i in tqdm(range(0, len(X), chunk_s), desc="Creating map..."):
        chunk = X[i:i + chunk_s]
        predict_map[i:i + chunk_s] = model.predict(chunk)
    predict_map = predict_map.reshape((s_x, s_y))
    print("Map is done.")
    
    out_img = images[0]
    out_img['array'] = predict_map
    save_data_tif(out_img, DEFAULT_PATH['output'] + name, color_palette)