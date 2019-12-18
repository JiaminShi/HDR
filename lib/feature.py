import numpy as np

from skimage.feature import (corner_harris, corner_peaks, plot_matches, BRIEF, match_descriptors)
from skimage.transform import warp, ProjectiveTransform
from skimage.color import rgb2gray
from skimage.measure import ransac

def produceMatches(imL1, imR1, panoramas = False, overlap_size = None):
    
    imL = imL1.copy()  
    imR = imR1.copy()
      
    if (panoramas):
        if (overlap_size == None):
            overlap_size = int(imL.shape[1] * 0.4)
        imL[:,:-overlap_size,:] = 0
        imR[:,overlap_size:,:] = 0 
    
    imLgray = rgb2gray(imL)
    imRgray = rgb2gray(imR)
    

    keypointsL = corner_peaks(corner_harris(imLgray), threshold_rel=0.001, min_distance=10)
    keypointsR = corner_peaks(corner_harris(imRgray), threshold_rel=0.001, min_distance=10)

    extractor = BRIEF()

    extractor.extract(imLgray, keypointsL)
    keypointsL = keypointsL[extractor.mask]         
    descriptorsL = extractor.descriptors

    extractor.extract(imRgray, keypointsR)
    keypointsR = keypointsR[extractor.mask]
    descriptorsR = extractor.descriptors

    matchesLR =  match_descriptors(descriptorsL, descriptorsR, cross_check=True)

    src = []
    dst = []
    for coord in matchesLR:
        src.append(keypointsL[coord[0]])  
        dst.append(keypointsR[coord[1]])
    src = np.array(src)
    dst = np.array(dst)

    src_c = src.copy()
    dst_c = dst.copy()
    src_c[:,1] = src[:,0]
    src_c[:,0] = src[:,1]
    dst_c[:,1] = dst[:,0]
    dst_c[:,0] = dst[:,1]

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src_c, dst_c), ProjectiveTransform, min_samples=4,
                               residual_threshold=8, max_trials=250)
    
    return (matchesLR, model_robust, inliers)