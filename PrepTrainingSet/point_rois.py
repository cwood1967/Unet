
import numpy as np
from read_roi import read_roi_file, read_roi_zip 
import tifffile
import scipy.ndimage as nd

def marks_from_roizip(roizip):
    # rois is a dictionary of roi data, key is name of roi
    rois = read_roi_zip(roizip)
    
    # want to get a list/array of x, y, z
    
    marks = list()
    for k, v in rois.items():
        x = v['x'][0]
        y = v['y'][0]
        z = v['position']['slice']
        marks.append([x, y, z,])
        
    return np.stack(marks)
            
            
def marks_to_mask(image, marks, blur=None):
    shape = image.shape
    mask = np.zeros((shape[0], shape[2], shape[3]), dtype=np.float32)
    for m in marks:
        mx = m[0]
        my = m[1]
        mz = m[2]
        mask[mz, my, mx] = 1
    
    if blur is not None:
        mask = nd.gaussian_filter(mask, blur)
        
    return mask

def points_to_mask(roiszip, image, blur):
    marks = marks_from_roizip(roiszip)
    mask = marks_to_mask(image, marks, blur=blur)
    return mask

    
#roifile = "/Volumes/core/micro/asa/fgm/smc/20190919_Screen/cjw_training/RoiSet_training_002.zip"
#image =  tifffile.imread("/Volumes/core/micro/asa/fgm/smc/20190919_Screen/cjw_training/training_002.tif")


