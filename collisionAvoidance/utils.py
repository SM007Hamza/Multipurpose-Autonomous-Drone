import numpy as np
from matplotlib import c, cm

import cv2

# Colormap wrapper


def applyColorMap(img, cmap):
    colormap = cm.get_cmap(cmap)
    colored = colormap(img)
    return np.float32(cv2.cvtColor(np.uint8(colored*255), cv2.COLOR_RGBA2BGR))/255.

# Convolutional wrapper 2 dimensional


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)
