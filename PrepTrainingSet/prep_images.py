import numpy as np
import os
import tifffile
import make_patches_3d
import point_rois
from matplotlib import pyplot as plt

def normalize(image, axis=0):
    imax = image.max(axis=axis, keepdims=True)
    imin = image.min(axis=axis, keepdims=True)
    return (image - imin)/(imax - imin)
    
def prep_image(imagefile, roifile, patch_size=256, blur=(1,2,2),
               cat_axis=1, norm_axis=(0,2,3)):
    image = tifffile.imread(imagefile)
    image = normalize(image, axis=norm_axis)
    print(image.shape)
    mask = point_rois.points_to_mask(roifile, image, blur)
    mask = np.expand_dims(mask, 1)
    mask = normalize(mask, axis=norm_axis)
    p0 = np.concatenate([image, mask], axis=cat_axis)
    p1 = make_patches_3d.make_patches(p0, patch_shape=(8,256,256))
    
    hv = np.where(p1[:,:,:,:,2].max(axis=(1,2,3)) > .5)[0]
    print("Val slice", len(hv), hv[-10:])
    pv = p1[hv[-10:]].copy()
    pt = np.delete(p1, hv[-10:], axis=0)
    a1 = make_patches_3d.augment_rotate(pt, 30)
    print(a1.shape, pv.shape, pt.shape)
    return a1, pv

datadir = "/n/core/micro/asa/fgm/smc/20190919_Screen/cjw_training/"
dimages = [{'imagefile':datadir + "training_002.tif",
            'roifile':datadir + "RoiSet_training_002.zip"},
           {'imagefile':datadir + "training_001.tif",
            'roifile':datadir + "RoiSet_training_001.zip"},
           {'imagefile':datadir + "training_003.tif",
            'roifile':datadir + "RoiSet_training_003.zip"},
           {'imagefile':datadir + "training_005.tif",
            'roifile':datadir + "RoiSet_training_005.zip"}
           ]

print("Running ...")

prep_list = list()
val_list = list()
for d in dimages:
    prepped, validation = prep_image(d['imagefile'], d['roifile'])
    print(d)
    prep_list.append(prepped)
    val_list.append(validation)

res = np.concatenate(prep_list, axis=0)
vres = np.concatenate(val_list, axis=0)
print(res.shape, vres.shape)
np.save(datadir + "augmented_b.npy", res)
np.save(datadir + "validation_b.npy", vres)