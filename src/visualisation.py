# %matplotlib inline
from matplotlib import pyplot as plt
from skimage.transform import resize


def plot_data(data, mask=None):
    data = resize(data, (1000, 1000), preserve_range=True, anti_aliasing=True)
    plt.style.use('dark_background')
    if mask is not None:
        mask = resize(mask, (1000, 1000), preserve_range=True, anti_aliasing=False)
        masked_data = data * (mask == 1)
        plt.imshow(masked_data, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    plt.show()