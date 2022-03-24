import matplotlib.pyplot as plt
from skimage import io, filters, util, restoration, morphology
import os
import time
import numpy as np


DATA_DIR = r'D:\Researchdata\ZY1'
# DATA_DIR = r'C:\Users\chen2\Documents\Research Project\ZY1'
THRESH = []

def FindMask(file_name, img_raw, paras_sharp, paras_rb, paras_otsu, paras_hys, paras_hat, paras_hist = [1024, 10]):
    # img_seed = np.copy(img_raw)
    # img_seed[1:-1, 1:-1] = img_raw.max()
    # img_mask = img_raw
    # img_filled = morphology.reconstruction(img_seed, img_mask, method='erosion')
    img_filled = morphology.closing(img_raw, morphology.disk(6))
    # img_filled = morphology.diameter_closing(img_raw, diameter_threshold = 6)
    # img_filled = morphology.opening(img_filled, morphology.disk(1))
    img_sharp = filters.unsharp_mask(img_filled, 
                                        radius = paras_sharp[0], 
                                        amount = paras_sharp[1],
                                        preserve_range = True)
    bg = restoration.rolling_ball(img_sharp, radius = paras_rb[0])
    bg_normal = util.img_as_int(filters.rank.mean(util.img_as_int(bg.astype(int)), selem=morphology.disk(paras_rb[1])))
    img_bg_reduced = img_sharp - bg_normal
    img_bg_reduced[img_sharp < bg_normal] = 0
    thresholds = filters.threshold_multiotsu(img_bg_reduced, classes = 4)
    img_otsu = np.digitize(img_bg_reduced, bins = thresholds)
    hist, bins = np.histogram(img_bg_reduced.ravel(), bins=paras_hist[0])
    inds = [np.where(bins > thr)[0][0] for thr in thresholds]
    diff = np.abs(hist[[i + paras_hist[1] for i in inds]] - hist[[i - paras_hist[1] for i in inds]])
    thresh = thresholds[np.argmin(diff)]
    if len(THRESH):
        if np.abs(thresh - THRESH[-1]) > 0.1 * THRESH[-1]:
            idx = np.argmin(np.abs(thresholds - THRESH[-1]))
            thresh = thresholds[idx]
    THRESH.append(thresh)
    # thresh = [thr for thr in thresholds if thr >= paras_otsu][0]
    low = thresh * paras_hys
    high = thresh
    img_hyst = filters.apply_hysteresis_threshold(img_bg_reduced, low, high).astype(int)
    res = morphology.white_tophat(img_hyst, morphology.disk(paras_hat))
    img_res = img_hyst - res.astype(int)

    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (10, 10))
    ax[0, 0].imshow(img_raw, cmap = 'gray')
    ax[0, 0].set_title('Raw')
    ax[0, 1].imshow(img_sharp, cmap = 'gray')
    ax[0, 1].set_title('Sharpened')
    ax[0, 2].imshow(img_bg_reduced, cmap = 'gray')
    ax[0, 2].set_title('BG reduced')
    ax[1, 0].imshow(img_otsu, cmap = 'jet')
    ax[1, 0].set_title('OTSU')
    ax[1, 1].imshow(img_hyst, cmap = 'gray')
    ax[1, 1].set_title('Hysteresis')
    ax[1, 2].imshow(img_res, cmap = 'gray')
    ax[1, 2].set_title('Tophat')
    ax[2, 0].hist(img_bg_reduced.ravel(), bins = paras_hist[0])
    for thr, dif in zip(thresholds, diff):
        if thr == thresh:
            ax[2, 0].axvline(thr, color = 'b', label = '%d-%d' % (thr, dif))
        else:
            ax[2, 0].axvline(thr, color = 'r', label = '%d-%d' % (thr, dif))
    ax[2, 0].legend()
    ax[2, 0].set_title('Hist: %d' % (thresh))
    ax[2, 1].imshow(bg_normal, cmap = 'gray')
    ax[2, 1].set_title('BG_normal')
    ax[2, 2].imshow(res, cmap = 'gray')
    ax[2, 2].set_title('Res')

    for a in ax.ravel():
        a.axis('off')
    
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches = 'tight')
    # plt.show()
    plt.close()

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
                    t0 = time.time()
                    FindMask(file_name, raw_img, [10, 2], [100, 30], 2500, .9, 2)
                    t1 = time.time()
                    print(t1 - t0)
    print("DONE")