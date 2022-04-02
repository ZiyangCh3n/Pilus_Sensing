import matplotlib.pyplot as plt
import numpy as np
from codes.find_mask import T_START
from skimage import io, restoration, filters, util, morphology
import os, sys
import time

ARG = sys.argv
YFP = 1 # YFP channel is the 2nd
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data', ARG[1])
MASK_DIR = os.path.join(DATA_DIR, 'mask')
IMG_DIR = os.path.join(DATA_DIR, [folder for folder in os.listdir(DATA_DIR) if folder.endswith('tiff')][0])
T_START = time.asctime(time.localtime(time.time()))

def GetMasked(dir_img, dir_mask, filename, timepoint, paras_rb = [150, 60]):
    img_stack = io.imread(dir_img)
    img_raw = img_stack[timepoint, ..., YFP]
    mask = np.bool_(io.imread(dir_mask)) * 1
    bg = restoration.rolling_ball(img_raw, radius = paras_rb[0])
    bg_normal = util.img_as_uint(filters.rank.mean(util.img_as_uint(bg.astype(int)), selem = morphology.disk(paras_rb[1])))
    img_bg_reduced = img_raw - bg_normal
    img_bg_reduced[img_raw < bg_normal] = 0
    img_masked = img_bg_reduced * mask
    
    x = [img_raw.ravel(), bg_normal.ravel(), img_bg_reduced.ravel()]
    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    ax[0, 0].imshow(img_raw, cmap = 'gray')
    ax[0, 0].set_title('YFP channel raw')
    ax[0, 1].imshow(mask, cmap = 'gray')
    ax[0, 1].set_title('Mask')
    ax[0, 2].imshow(img_bg_reduced, cmap = 'gray')
    ax[0, 2].set_title('BG reduced')
    ax[1, 0].imshow(img_masked, cmap = 'gray')
    ax[1, 0].set_title('Masked')
    ax[1, 1].imshow(bg_normal, cmap = 'gray')
    ax[1, 1].set_title('Background') 
    ax[1, 2].hist(np.array(x).T, 200, density = True, histtype = 'step', stacked = True, fill = False, label = ['raw', 'bg', 'bg_reduced'])
    ax[1, 2].legend(loc = 'upper left')
    ax[1, 2].set_title('Hitogram - gray value')
    for a in ax.ravel()[:-1]:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    return img_masked

if __name__ == '__main__':
    t0 = time.time()
    if not os.path.exists(os.path.join(DATA_DIR, 'YFP_ref')):
        os.mkdir(os.path.join(DATA_DIR, 'YFP_ref'))
    for parent_img, dir_img, file_img in os.walk(IMG_DIR):
        for f_img in file_img:
            dir_img = os.path.join(parent_img, f_img)
            for parent, dir, file in os.walk(MASK_DIR):
                for f in file:
                    if f_img[:10] == f[:10]:
                        dir_mask = os.path.join(parent, f)
                        timepoint = int(f[11:13])
                        file_name = os.path.join(DATA_DIR, 'YFP_ref', f)
                        img_masked = GetMasked(dir_img, dir_mask, file_name, timepoint)
    t1 = time.time()
    with open(os.path.join(DATA_DIR, 'log'), 'a') as log
        log.write('-'*10 + 'Calculate YFP' + '-'*10 + '\n')
        log.write('Started at: ' + T_START + '\n')
        log.write('DATA_DIR: ' + DATA_DIR + '\n')
        log.write('Totaltime in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')

