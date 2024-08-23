import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def default_transform(x):
        x = np.array(x).transpose(1, 2, 0)
        x = (x - x.min()) / (x.max() - x.min())
        return x

def image_grid(images, labels=None, idx_to_class=None, cmap=None, transform=default_transform, **fig_kw):
        ncols = min(8, len(images))
        nrows = ceil(len(images)/ncols)
        fig, axs = plt.subplots(nrows, ncols, **fig_kw)
        if len(images) == 1: axs = np.array([axs])
        axs = axs.flatten()
        for ax in axs:  ax.set_axis_off()        
        for i, image in enumerate(images):
                image = transform(image)
                axs[i].imshow(image, cmap=cmap)
                if labels is not None:
                        label = idx_to_class[labels[i].item()] if idx_to_class else labels[i].item()
                        axs[i].set_title(label)
