# This is for the purpose of getting fluorescence values for the images in the dataset using masks
# put this in the analysis folder with the slurm

import numpy as np
# from  skimage import io, filters, util, restoration, morphology, exposure, segmentation
from skimage.restoration import rolling_ball
from skimage.util import img_as_uint
from skimage.filters import rank
from skimage.measure import label, regionprops
from skimage.morphology import disk
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as mpatches
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

### Calculate fluorescence
def CalcFluo(mc_raw, yfp_raw, img_name):
    if os.path.exists(os.path.join(FLUO_DIR, img_name)):
        return 0
    mc_bg = Preprocess(mc_raw)
    yfp_bg = Preprocess(yfp_raw)
    # find mask

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
                    slice_mc = np.arange(CHANNEL['mcherry'], img_stack.shape[0], 3)
                    slice_yfp = np.arange(CHANNEL['YFP'], img_stack.shape[0], 3)
                    for i in range(len(slice)):
                        mc_raw = img_stack[slice_mc[i], ...]
                        yfp_raw = img_stack[slice_yfp[i], ...]
                        img_name = ('_'.join((os.path.splitext(f)[0], str(i).zfill(2))) + '.png')
                        CalcFluo(mc_raw, yfp_raw, img_name)
                    end_time = time.time()
                    with open(os.path.join(ANALYSIS_DIR, 'progress', 'CF_%d-%d.txt' % (START, STOP)), 'a') as w:
                        w.write("Finished: %s, Time: %.2f s\n" % (f, end_time - start_time))
            LOC += 1
    t1 = time.time()
    with open(os.path.join(ANALYSIS_DIR, 'progress', 'CF_%d-%d.txt' % (START, STOP)), 'a') as w:
        w.write("Total time: %.2f min\n" % ((t1 - t0) / 60))
                    