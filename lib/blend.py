import numpy as np
from skimage.color import rgb2gray

def boundaryDT(image):
    imgGray = rgb2gray(image)
    distanceMap = np.zeros(imgGray.shape)
    rows = imgGray.shape[0]
    cols = imgGray.shape[1]       
    totalDist = 0.
        
    for c in range(cols):
        for r in range(rows):
            distanceMap[r, c] = min(r , c, (rows - r), (cols - c))
            totalDist += distanceMap[r, c]
    
    return distanceMap / totalDist  


def alpha(distMap1, distMap2, refHeight, refWidth):
    ret = np.zeros((2, refHeight, refWidth))
    sumMap = distMap1 + distMap2
    sumMap = np.where(sumMap == 0, 100000, sumMap)
    ret[0,:,:] = distMap1 / sumMap
    ret[1,:,:] = distMap2 / sumMap
    return ret