import matplotlib.pyplot as plt
import numpy as np
import math

from PIL import Image
from skimage.color import rgb2gray

def img2nparray(img):
    # in case of img is RGB
    if img.ndim == 3:
        #print('input image is RGB!')
        img = rgb2gray(img)
        img = img*255
        arr = np.array(img, dtype = int)
        
    else:
        #print('input image is not RGB!')
        arr = np.array(img, dtype = int)
    return arr

def imgSeg(arr, seg_num):
    max_arr = np.max(arr)
    div = math.ceil(max_arr/seg_num)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = (arr[i][j]//div)*div
            
    return arr

def nparray2perseus(arr):
    f_perseus = 'temp.txt'
    temp = open(f_perseus, mode='w', encoding='utf-8')

    # write dimension of image in first line 
    data = ('%d' % arr.ndim)
    temp.write(data)

    # write number of row and column consecutively
    for i in range(arr.ndim):
        data = ('\n%d' % arr.shape[i])
        temp.write(data)
    
    # write pixel value from bottom to top
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p_val=arr[arr.shape[0]-i-1][j]
            data = ('\n%d' % p_val)
            temp.write(data)
    temp.close()
    return f_perseus


def img2perseus(file, seg_num = None):
    img = plt.imread(file)
    arr = img2nparray(img)
    if seg_num != None:
        arr = imgSeg(arr, seg_num)
    f_perseus = nparray2perseus(arr)
    return f_perseus