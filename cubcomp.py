import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import gudhi as gd
import numpy as np

def getPH(image):
    image = plt.imread(image)
    gray = rgb2gray(image)
    gray = gray*255
    gray = np.array(gray, dtype = int)
    height, width = gray.shape
    bitmap = []
    for i in range(0, height):
        for j in range(0, width):
            norm = gray[i][j]
            bitmap.append(norm)

    bcc = gd.CubicalComplex(top_dimensional_cells = bitmap, dimensions=[height, width])
    return bcc.persistence()