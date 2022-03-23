import matplotlib.pyplot as plt
from skimage import io, filters, util, restoration, morphology
import os
import numpy as np


DATA_DIR = r'D:\Researchdata\ZY1'

def FindMask(file_name, raw_img, paras_sharp, paras_rb, paras_hys, paras_hat):
    sharpen_img = filters.unsharp_mask(raw_img, 
                                        radius = paras_sharp[0], 
                                        amount = paras_sharp[1],
                                        preserve_range = True)
    bg = restoration.rolling_ball(sharpen_img, radius = paras_rb[0])
    footprint = morphology.disk(paras_rb[1])
    normal_bg = util.img_as_int(filters.rank.mean(util.img_as_int(bg.astype(int)), selem=footprint))
    bg_reduced_img = sharpen_img - normal_bg
    thresholds = filters.threshold_multiotsu(bg_reduced_img, classes = 4)
    otsu_img = np.digitize(bg_reduced_img, bins = thresholds)
    thresh = [thr for thr in thresholds if thr >= 2500][0]
    # thresh = filters.threshold_otsu(bg_reduced_img)
    # otsu_img = (bg_reduced_img > thresh).astype(int)
    low = thresh * paras_hys
    high = thresh
    # lowt = (bg_reduced_img >= low).astype(int)
    # hight = (bg_reduced_img >= high).astype(int)
    hyst_img = filters.apply_hysteresis_threshold(bg_reduced_img, low, high).astype(int)
    footprint = morphology.disk(paras_hat)
    res = morphology.white_tophat(hyst_img, footprint)
    res_img = hyst_img - res.astype(int)

    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 10))
    ax[0, 0].imshow(raw_img, cmap = 'gray')
    ax[0, 0].set_title('Raw')
    ax[0, 1].imshow(sharpen_img, cmap = 'gray')
    ax[0, 1].set_title('Sharpened')
    ax[0, 2].imshow(bg_reduced_img, cmap = 'gray')
    ax[0, 2].set_title('BG reduced')
    ax[1, 0].imshow(otsu_img, cmap = 'jet')
    ax[1, 0].set_title('OTSU')
    ax[1, 1].imshow(hyst_img, cmap = 'gray')
    ax[1, 1].set_title('Hysteresis')
    ax[1, 2].imshow(res_img, cmap = 'gray')
    ax[1, 2].set_title('Tophat')

    for a in ax.ravel():
        a.axis('off')
    
    plt.tight_layout()
    # plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_DIR, 'mask')):
        os.mkdir(os.path.join(DATA_DIR, 'mask'))
    for parent, dir, file in os.walk(DATA_DIR):
        if( 'tiff' in parent):
            for f in file:
                raw_stack = io.imread(os.path.join(parent, f))
                for i in range(raw_stack.shape[0]):
                    raw_img = raw_stack[i, ..., 2]
                    file_name = os.path.join(DATA_DIR, 'mask', ('_'.join((os.path.splitext(f)[0], str(i))) + '.png'))
                    FindMask(file_name, raw_img, [10, 2], [100, 60], .9, 1)
