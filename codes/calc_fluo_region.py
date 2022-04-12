import numpy as np
from  skimage import io
from skimage.restoration import rolling_ball
from skimage.util import img_as_uint
from skimage.filters import rank
from skimage.measure import label, regionprops
from skimage.morphology import disk
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from os import path, walk, mkdir, listdir, getcwd
import time, sys
import pandas as pd

ARG = sys.argv
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}
DATA_DIR = path.join(path.dirname(getcwd()), 'data', ARG[1])
T_START = time.asctime(time.localtime(time.time()))
START = int(ARG[2])
if not int(ARG[3]): # set to 0 to run through all files
    ARG[3] = 10000
STOP = int(ARG[2]) + int(ARG[3]) - 1

def GetImageBgHist(img, channel, tp, paras_rb = [50, 50]):
    img_raw = img[tp, ..., CHANNEL[channel]]
    bg = rolling_ball(img_raw, radius = paras_rb[0])
    bg_normal = img_as_uint(rank.mean(img_as_uint(bg.astype(int)), selem = disk(paras_rb[1])))
    img_bg_reduced = img_raw - bg_normal
    img_bg_reduced[img_raw < bg_normal] = 0
    # img_masked = img_bg_reduced * mask
    
    x = [img_raw.ravel(), bg_normal.ravel(), img_bg_reduced.ravel()]
    return img_bg_reduced, bg_normal, np.array(x).T

def GetMaskRegion(mask, centroid0, ids0, area_thr = 100):
    mask_labeled0 = label(mask, background=0, connectivity=2)
    props0 = regionprops(mask_labeled0)
    for region in props0:
        if region.area < area_thr:
            mask[region.coords[:, 0], region.coords[:, 1]] = 0
    # after deleting small objects, do labeling again
    mask_labeled = label(mask, background=0, connectivity=2)
    props = regionprops(mask_labeled)
    centroid = np.array([region.centroid for region in props])
    ids = np.array([mask_labeled[region.coords[0][0], region.coords[0][1]] for region in props])
    if len(centroid0):
        dist = cdist(centroid, centroid0, metric='euclidean')
        match = linear_sum_assignment(dist)
        for r, r0 in zip(match[0], match[1]):
            region = props[r]
            mask_labeled[region.coords[:, 0], region.coords[:, 1]] = ids0[r0]
        # for r, r0 in zip(match[0], match[1]):
        #     mask_labeled[mask_labeled == ids[r]] = ids0[r0]
        ids = ids0[match[1]]
        centroid = centroid[match[0]]
    
    return mask_labeled, centroid, ids

# Calculate fluo of each position-timepoint img
def CalcFluoByRegion(img, mask, centroid0, ids0, filename, tp):
    yfp_bg_reduced, yfp_bg, yfp_hist = GetImageBgHist(img, 'YFP', tp)
    mc_bg_reduced, mc_bg, mc_hist = GetImageBgHist(img, 'mcherry', tp)
    mask_labeled, centroid, ids = GetMaskRegion(mask, centroid0, ids0)
    props = regionprops(mask_labeled)
    stats = {}

    # visualization
    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 15))
    c0, c1 = 'YFP', 'mcherry'
    ax[0, 0].imshow(img[tp, ..., CHANNEL[c0]], cmap = 'gray')
    ax[0, 0].set_title("%s image raw" % c0)
    ax[0, 0].axis('off')
    ax[0, 1].imshow(yfp_bg_reduced, cmap = 'gray')
    ax[0, 1].set_title("%s BG reduced" % c0)
    ax[0, 1].axis('off')
    ax[0, 2].hist(yfp_hist, 200, density = True, histtype = 'step', stacked = False, fill = False, label = ['raw', 'bg', 'bg_reduced'])
    ax[0, 2].legend(loc = 'upper left')
    ax[0, 2].set_title("%s histogram" % c0)
    
    ax[1, 0].imshow(img[tp, ..., CHANNEL[c1]], cmap = 'gray')
    ax[1, 0].set_title("%s image raw" % c1)
    ax[1, 0].axis('off')
    ax[1, 1].imshow(mc_bg_reduced, cmap = 'gray')
    ax[1, 1].set_title("%s BG reduced" % c1)
    ax[1, 1].axis('off')
    ax[1, 2].hist(mc_hist, 200, density = True, histtype = 'step', stacked = False, fill = False, label = ['raw', 'bg', 'bg_reduced'])
    ax[1, 2].legend(loc = 'upper left')
    ax[1, 2].set_title("%s histogram" % c1)
    
    ax[2, 1].imshow(yfp_bg, cmap = 'gray')
    ax[2, 1].set_title("%s BG" % c0)
    ax[2, 1].axis('off')
    ax[2, 2].imshow(mc_bg, cmap = 'gray')
    ax[2, 2].set_title("%s BG" % c1)    
    ax[2, 2].axis('off')
    
    ax[2, 0].imshow(mask, cmap = 'gray')
    ax[2, 0].set_title("Mask Region")
    ax[2, 0].axis('off')
        
    for region in props:
        ident = mask_labeled[region.coords[0][0], region.coords[0][1]]
        ax[2, 0].text(region.centroid[1], region.centroid[0], str(ident), c = 'r')
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                                  fill = False, edgecolor = 'y', linewidth = 2)
        ax[2, 0].add_patch(rect)
        mask_region = np.zeros(mask.shape)
        mask_region[region.coords[:, 0], region.coords[:, 1]] = 1
        row = {'Position': filename, 'Area': region.area, 
               'YFP intensity total': np.sum(yfp_bg_reduced * mask_region),
               'YFP background': np.sum(yfp_bg * mask_region), 
               'mCherry intensity total': np.sum(mc_bg_reduced * mask_region),
               'mCherry background': np.sum(mc_bg * mask_region)}
        stats[ident] = row
    
    
    plt.tight_layout()
    plt.savefig(path.join(data_dir, 'fluo_ref', filename), bbox_inches = 'tight')
    plt.close()
    stats = pd.DataFrame.from_dict(stats, orient = 'index').reset_index().rename(columns = {'index': 'timepoint'})
    return centroid, ids, stats

def CalcFluoMain(img, mask_dir, position):
    centroid0 = []
    ids0 = []
    # dict_df = {}
    
    for parent, folder, file in walk(mask_dir):
        for f in file:
            tp_mask = int(f.removesuffix('.png').split('_')[3])
            pos_mask = int(f.removesuffix('.png').split('_')[2])
            if pos_mask == position:
                # if path.exists(path.join(data_dir, 'fluo_ref', f)):
                #     continue
                mask = io.imread(path.join(parent, f))
                centroid0, ids0, stats = CalcFluoByRegion(img, mask, centroid0, ids0, f, tp_mask)
                try:
                    df = pd.concat([df, stats])
                except NameError:
                    df = stats    

        
    # for key, val in zip(dict_df.keys(), dict_df.values()):
    #     dict_df[key] = pd.DataFrame.from_dict(val, orient = 'index').reset_index().rename(columns = {'index': 'timepoint'})
        df.to_csv(path.join(data_dir, 'fluorescence.csv'), index = False, mode = 'a',
              header = not path.exists(path.join(data_dir, 'fluorescence.csv')))
        # return df
                
if __name__ == '__main__':
    LOC = 0
    for parent_m, folder_m, file_m in walk(DATA_DIR):
        if 'ref.txt' in file_m:
            flag = False
            t0 = time.time()
            data_dir = parent_m
            imgfolder_dir = path.join(data_dir, [folder for folder in listdir(data_dir) if folder.endswith('tiff')][0])
            maskfolder_dir = path.join(data_dir, 'mask')
            if not path.exists(path.join(data_dir, 'fluo_ref')):
                mkdir(path.join(data_dir, 'fluo_ref'))
            for parent_img, folder_img, file_img in walk(imgfolder_dir):
                for f in file_img:
                    if START <= LOC <= STOP:
                        flag = True
                        img_dir = path.join(parent_img, f)
                        img = io.imread(img_dir)
                        pos = int(f.removesuffix('.tiff').split('_')[2])
                        CalcFluoMain(img, maskfolder_dir, pos)
                    LOC += 1
            t1 = time.time()
            if flag:
                with open(path.join(data_dir, 'log'), 'a') as log:
                    log.write('-'*10 + 'Fluorescence' + '-'*10 + '\n')
                    log.write('Started at: ' + T_START + '\n')
                    log.write('DATA_DIR: ' + data_dir + '\n')
                    log.write('START: %d STOP: %d \n' % (START % 36, STOP % 36))
                    log.write('Totaltime in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')
                                
                        