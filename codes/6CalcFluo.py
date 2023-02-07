# This is for the purpose of getting fluorescence values for the images in the dataset using masks
# put this in the analysis folder with the slurm

import numpy as np
from  skimage import io, filters, util, restoration, morphology, exposure, segmentation
from skimage.restoration import rolling_ball
# from skimage.util import img_as_uint
from skimage.filters import rank
# from skimage.measure import label, regionprops
from skimage.morphology import disk
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import time, sys
import pandas as pd


# Get working directory
os.chdir(os.path.dirname(os.getcwd()))
MASK_DIR = os.path.join(os.getcwd(), 'masks', 'mask_out')
TIF_DIR = os.path.join(os.getcwd(), 'tiff')
ANALYSIS_DIR = os.path.join(os.getcwd(), 'analysis')
FLUO_DIR = os.path.join(os.getcwd(), 'fluo')

## Get arugments
ARG = sys.argv
START = int(ARG[1])
STOP = int(ARG[1]) + int(ARG[2])
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}

## Functions
### Remove background
def Preprocess(img_raw, r = 200):
    bg = rolling_ball(img_raw, radius = r)
    bg_normal = rank.mean(bg, footprint= disk(r))
    img_bg_removed = img_raw - bg_normal
    img_bg_removed[img_raw < bg_normal] = 0
    return (img_bg_removed)

### Calculate fluorescence of each slice
def CalcFluoSlice(mc_raw, yfp_raw, img_name):
    if os.path.exists(os.path.join(FLUO_DIR, img_name)):
        return 0
    # find mask
    for parent, folder, file in os.walk(MASK_DIR):
        if 'ipy_checkpoints' in parent:
            continue
        file.sort()
        for f in file:
            if f.endswith('.png'):
                if f.split('_', 1)[1] == img_name:
                    mask = io.imread(os.path.join(parent, f))
                    break

    # calculate fluorescence
    yfp_nbg = Preprocess(yfp_raw)
    mc_nbg = Preprocess(mc_raw)
    area = np.sum(mask, dtype=np.uint64)
    yfp = np.sum(yfp_nbg * mask, dtype=np.uint64) / area
    mc = np.sum(mc_nbg * mask, dtype=np.uint64) / area
    yfp0 = np.sum(yfp_raw * mask, dtype=np.uint64) / area
    mc0 = np.sum(mc_raw * mask, dtype=np.uint64) / area
    yfp_bg = yfp0 - yfp
    mc_bg = mc0 - mc

    stats = pd.DataFrame({'filename': img_name, 'yfp': [yfp], 'mc': [mc], 'yfp0': [yfp0], 'mc0': [mc0], 'yfp_bg': [yfp_bg], 'mc_bg': [mc_bg], 'area': [area]})

    # image output
    fig, ax = plt.subplots(2, 3, figsize = (15, 10))
    imgs = [yfp_raw, yfp_nbg, yfp_nbg * mask, mc_raw, mc_nbg, mc_nbg * mask]
    titles = ['YFP raw', 'YFP bg removed', 'YFP masked', 'mCherry raw', 'mCherry bg removed', 'mCherry masked']
    ax = ax.ravel()
    for i in range(len(ax)):
        ax[i].imshow(imgs[i], cmap = 'gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FLUO_DIR, img_name), bbox_inches = 'tight')
    plt.close()
    
    stats.to_csv(os.path.join(ANALYSIS_DIR, 'fluo', '%d.csv' % (STOP / (STOP - START))), mode = 'a', header = False, index = False)


## Main
if __name__ == '__main__':
    LOC = 0
    t0 = time.time()
    for parent, folder, file in os.walk(TIF_DIR):
        if 'ipy_checkpoints' in parent:
            continue
        file.sort()
        for f in file:
            if START <= LOC < STOP:
                start_time = time.time()
                if f.endswith('.tif'):
                    img_dir = os.path.join(parent, f)
                    img_stack = io.imread(img_dir)
                    img_stack = img_stack.astype(np.uint16)
                    slice_mc = np.arange(CHANNEL['mcherry'], img_stack.shape[0], 3)
                    slice_yfp = np.arange(CHANNEL['YFP'], img_stack.shape[0], 3)
                    for i in range(len(slice_mc)):
                        mc_raw = img_stack[slice_mc[i], ...]
                        yfp_raw = img_stack[slice_yfp[i], ...]
                        img_name = ('_'.join((os.path.splitext(f)[0], str(i).zfill(2))) + '.png')
                        CalcFluoSlice(mc_raw, yfp_raw, img_name)
                    end_time = time.time()
                    with open(os.path.join(ANALYSIS_DIR, 'progress', 'CF_%d-%d.txt' % (START, STOP)), 'a') as w:
                        w.write("Finished: %s, Time: %.2f s\n" % (f, end_time - start_time))
            LOC += 1
    t1 = time.time()
    with open(os.path.join(ANALYSIS_DIR, 'progress', 'CF_%d-%d.txt' % (START, STOP)), 'a') as w:
        w.write("Total time: %.2f min\n" % ((t1 - t0) / 60))
                    