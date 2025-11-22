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


def plot_confusion(matrix):
    # TODO: setup plot
    plt.clf()

    plt.imshow(matrix)
    plt.colorbar()
    plt.show()


def spyder_eye(datasets, ml_sets):
    if len(datasets) != ml_sets:
        Exception("Lens of datasets != ml_sets")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(len(datasets)):
        # draw 1st line: conf_matrix
        ds = datasets[i]
        heatmap = ml_sets[i]['cf_matrix']
        heatmap = heatmap / heatmap.sum(axis=1)     # normalise it.

        percent = ml_sets[i].get('percent', '')
        mask_mode = ds.get('mask_mode', '')
        rows = ml_sets[i].get('r', '')
        resize_val = ml_sets[i].get('resize', '')
        im = axes[0, i].imshow(heatmap, cmap='viridis')
        
        rows, cols = heatmap.shape
        for r in range(rows):
            for c in range(cols):
                if heatmap[r, c] > 0.3:
                    axes[0, i].text(c, r, f"{heatmap[r, c]:.2f}", ha='center', va='center')

        title = f"{percent}*{mask_mode}"
        axes[0, i].set_title(title)
        plt.colorbar(im, ax=axes[0, i])

        # draw 2nd line: hist of classes

        classes = range(ds['classes'].size)
        axes[1, i].bar(classes, ds['classes'])
        axes[1, i].set_xlabel('Class')
        axes[1, i].set_ylabel('Count')
    plt.tight_layout()
    plt.show()