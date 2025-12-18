# Module 2: create map by predict on weights
import joblib
import numpy as np
from tqdm import tqdm
import time

from utils import save_tif, color_palette, DEFAULT_PATH


def create_map(images: dict, model, name: str, count_chunks=2, with_time=True):
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

    start_time = time.time()

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
    elapsed_time = round((time.time() - start_time) / 60, 1)
    
    out_img = images[0]
    out_img['array'] = predict_map
    save_tif(out_img, name, color_palette=color_palette)

    return name