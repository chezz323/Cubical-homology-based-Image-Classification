import numpy as np

def central_moment(img, order=1):
    x, counts= np.unique(img, return_counts=True)
    sumx = 0
    for idx in range(len(x)):
        sumx = sumx + x[idx]*counts[idx]
    mean = sumx/np.sum(counts)
    
    if order==1:
        return mean
    elif order>1:
        sumx=0
        for idx in range(len(x)):
            sumx = sumx + ((x[idx]-mean)**order)*counts[idx]
        return sumx/np.sum(counts)
    else:
        return print("Error: Check input again")


    