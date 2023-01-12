# This is for the purpose of getting all the masks for the images in the dataset
# put this in the analysis folder with the slurm
import matplotlib.pyplot as plt
from skimage import io, filters, util, restoration, morphology, exposure, segmentation
import time
import numpy as np
import os
import sys

## Get working directory
os.chdir(os.path.dirname(os.getcwd())) # go up one directory to the data folder
MASK_DIR = os.path.join(os.getcwd(), 'masks')
TIF_DIR = os.path.join(os.getcwd(), 'tiff')
ANALYSIS_DIR = os.path.join(os.getcwd(), 'analysis')

## Get arguments
ARG = sys.argv
START = int(ARG[1])
STOP = int(ARG[1]) + int(ARG[2])
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}

## Functions for all masks
### local_threshold
def LocalThresh(img):
    try:
        block_size = 1001#1001
        local_thresh = filters.threshold_local(img, block_size)
        binary_local = img >= local_thresh
        return binary_local
    except Exception as e:
        return False
### global_otsu
def GlobalOtsu(img):
    try:
        global_thresh = filters.threshold_otsu(img)
        binary_global = img > global_thresh
        return binary_global
    except Exception as e:
        return False
### multi_otsu - aborted
def MultiOtsu(img):
    try:
        thresholds = filters.threshold_multiotsu(img, classes = 4)
        img_otsu = np.digitize(img, bins = thresholds)
        return (img_otsu, thresholds)
    except Exception as e:
        return False
### local_otsu
def LocalOtsu(img):
    try:
        footprint = morphology.disk(1001)#1001
        local_otsu = filters.rank.otsu(img, footprint)
        binary_local = img > local_otsu
        return binary_local
    except Exception as e:
        return False
### morpheACWE
def MorpheACWE(img):
    try:
        init_ls = segmentation.checkerboard_level_set(img.shape, 6)
        evolution = []
        def store_evolution_in(lst):
            def _store(x):
                lst.append(np.copy(x))
            return _store
        callback = store_evolution_in(evolution)
        ls = segmentation.morphological_chan_vese(img, num_iter = 200, init_level_set = init_ls, smoothing = 1, iter_callback = callback)
        return ls #200
    except Exception as e:
        return False
### Isodata
def Isodata(img_raw):
    try:
        thresh = filters.threshold_isodata(img_raw)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### Li
def Li(img_raw):
    try:
        thresh = filters.threshold_li(img_raw)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### Mean
def ThreshMean(img_raw):
    try:
        thresh = filters.threshold_mean(img_raw)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### saulova
def Sauvola(img_raw):
    try:
        thresh = filters.threshold_sauvola(img_raw, window_size = 25)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### niblack
def Niblack(img_raw):
    try:
        thresh = filters.threshold_niblack(img_raw, window_size = 25, k = .8)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### triangle
def Triangle(img_raw):
    try:
        thresh = filters.threshold_triangle(img_raw)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False
### yen
def Yen(img_raw):
    try:
        thresh = filters.threshold_yen(img_raw)
        binary = img_raw > thresh
        return binary
    except Exception as e:
        return False

## Functions for removing background
def Preprocess(img_raw, r = 200):
    # t0 = time.time()
    # img_filled = morphology.closing(img_raw, morphology.disk(6))
    # img_sharp = filters.unsharp_mask(img_filled, radius = 10, amount = 2, preserve_range = True)
    bg = restoration.rolling_ball(img_raw, radius = r)
    bg_normal = filters.rank.mean(bg, footprint = morphology.disk(r))
    img_bg_reduced = img_raw - bg_normal
    img_bg_reduced[img_raw < bg_normal] = 0
    # t1 = time.time()
    return (img_bg_reduced)

## The general function to write masks
def CreateMask(mask_dir, img_raw, img_name):
    if os.path.exists(os.path.join(mask_dir, 'preview', img_name)):
        return 0
    ori = img_raw #1
    img_raw_bg = Preprocess(img_raw)
    lt = LocalThresh(img_raw_bg) #2
    go = GlobalOtsu(img_raw_bg) #3
    lo = LocalOtsu(img_raw_bg) #4
    ma = MorpheACWE(img_raw_bg) #5
    iso = Isodata(img_raw_bg) #6
    li = Li(img_raw_bg) #7
    mn = ThreshMean(img_raw_bg) #8
    tr = Triangle(img_raw_bg) #9
    yn = Yen(img_raw_bg) #10
    sa = Sauvola(img_raw_bg) #11
    ni = Niblack(img_raw_bg) #12
    fig, ax = plt.subplots(3, 4, figsize = (9,9), dpi = 100)
    ax = ax.flatten()
    titles =  ['Original', 'LocalThresh', 'GlobalOtsu', 'LocalOtsu', 
               'MorpheACWE', 'Isodata', 'Li', 'Mean', 
               'Triangle', 'Yen', 'Sauvola', 'Niblack']
    imgs = [ori, lt, go, lo, 
            ma, iso, li, mn,
           tr, yn, sa, ni]
    for i in range(12):
        if type(imgs[i]) == bool and not imgs[i]:
            continue
        out_dir = os.path.join(mask_dir, 'masks_all', '_'.join((titles[i], img_name)))
        # print(out_dir)
        io.imsave(out_dir, util.img_as_ubyte(imgs[i]))
        if i == 12:
            ax[i].imshow(imgs[i], cmap='turbo')
        else:
            ax[i].imshow(imgs[i], cmap="gray")
        ax[i].set_axis_off()
        ax[i].set_title(titles[i], fontsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(mask_dir, 'preview', img_name), bbox_inches = 'tight')
    plt.close()

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
                    slice = np.arange(CHANNEL['mcherry'], img_stack.shape[0], 3)
                    for i in range(len(slice)):
                        img_raw = img_stack[slice[i], ...]
                        img_name = ('_'.join((os.path.splitext(f)[0], str(i).zfill(2))) + '.png')
                        CreateMask(MASK_DIR, img_raw, img_name)
                    end_time = time.time()
                    with open(os.path.join(ANALYSIS_DIR, 'progress_%d-%d.txt' % (START, STOP)), 'a') as w:
                        w.write("Finished: %s, Time: %.2f s\n" % (f, end_time - start_time))
            LOC += 1
    t1 = time.time()
    with open(os.path.join(ANALYSIS_DIR, 'progress_%d-%d.txt' % (START, STOP)), 'a') as w:
        w.write('TOTAL TIME: %.2f min' % ((t1 - t0) / 60))
                
