import numpy as np


def point_cloud(self, depth):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - self.cx) / self.fx, 0)
    y = np.where(valid, z * (r - self.cy) / self.fy, 0)
    return np.dstack((x, y, z))
