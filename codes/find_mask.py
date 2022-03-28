import matplotlib.pyplot as plt
from skimage import io, filters, util, restoration, morphology
import os
import time
import numpy as np
import sys

ARG = sys.argv
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data', ARG[1])
THRESH = []
log = open(os.path.join(DATA_DIR, 'log'), 'a')
log.write('-'*10 + 'Masking' + '-'*10 + '\n')
log.write('Started at: ' + time.asctime(time.localtime(time.time())) + '\n')
log.write('DATA_DIR: ' + DATA_DIR + '\n')

def FindMin(hist, bins, thresholds, window_width = 20):
    cumsum = np.cumsum(hist)
    half_width = int(window_width / 2)
    moving_average = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    moving_bins = bins[half_width : -half_width]
    inds = [np.where(moving_bins > thr)[0][0] for thr in thresholds]
    lb = [idx - half_width if idx > half_width else 0 for idx in inds]
    rb = [idx + half_width if idx + half_width < len(moving_bins) -1 else len(moving_bins) -1 for idx in inds]
    moving_diff = np.abs(np.diff(moving_average, 1))
    moving_diff_min = np.int32([np.min(moving_diff[a:b]) / 10 for a, b in zip(lb, rb)])
    min_inds_relative = [np.argmin(moving_diff[a:b]) for a, b in zip(lb, rb)]
    min_inds = np.sum([min_inds_relative, lb], axis = 0)
    bin_min = moving_bins[min_inds]
    return moving_diff_min, np.int32(bin_min)

def FindMask(file_name, img_raw, paras_close, paras_sharp, paras_rb, paras_hys, paras_hat, paras_hist = [1024, 10]):
    img_filled = morphology.closing(img_raw, morphology.disk(paras_close))
    # img_filled = morphology.diameter_closing(img_raw, diameter_threshold = 6)
    # img_filled = morphology.opening(img_filled, morphology.disk(1))
    img_sharp = filters.unsharp_mask(img_filled, 
                                        radius = paras_sharp[0], 
                                        amount = paras_sharp[1],
                                        preserve_range = True)
    bg = restoration.rolling_ball(img_sharp, radius = paras_rb[0])
    bg_normal = util.img_as_int(filters.rank.mean(util.img_as_int(bg.astype(int)), selem=morphology.disk(paras_rb[1])))
    # bg_normal = util.img_as_uint(filters.rank.mean(util.img_as_uint(bg.astype(int)), selem=morphology.disk(paras_rb[1])))
    img_bg_reduced = img_sharp - bg_normal
    img_bg_reduced = img_sharp - bg_normal
    img_bg_reduced[img_sharp < bg_normal] = 0
    thresholds = filters.threshold_multiotsu(img_bg_reduced, classes = 4)
    img_otsu = np.digitize(img_bg_reduced, bins = thresholds)
    hist, bins = np.histogram(img_bg_reduced.ravel(), bins=paras_hist[0])
    # inds = [np.where(bins > thr)[0][0] for thr in thresholds]
    # diff = np.abs(hist[[i + paras_hist[1] for i in inds]] - hist[[i - paras_hist[1] for i in inds]])
    diff, thresh_min = FindMin(hist, bins, thresholds)
    thresh = thresh_min[np.argmin(diff)]
    # thresh = thresholds[np.argmin(diff)]
    if len(THRESH):
        if np.abs(thresh - THRESH[-1]) > 0.4 * THRESH[-1]:
            idx = np.argmin(np.abs(thresholds - THRESH[-1]))
            thresh = thresholds[idx]
    THRESH.append(thresh)
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
    ax[0, 2].set_title('BG reduced, r = %d' % paras_rb[0])
    ax[1, 0].imshow(img_otsu, cmap = 'jet')
    ax[1, 0].set_title('OTSU')
    ax[1, 1].imshow(img_hyst, cmap = 'gray')
    ax[1, 1].set_title('Hysteresis')
    ax[1, 2].imshow(img_res, cmap = 'gray')
    ax[1, 2].set_title('Tophat')
    ax[2, 0].hist(img_bg_reduced.ravel(), bins = paras_hist[0])
    for thr, dif in zip(thresh_min, diff):
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
    t0 = time.time()
    if not os.path.exists(os.path.join(DATA_DIR, 'mask')):
        os.mkdir(os.path.join(DATA_DIR, 'mask'))
    for parent, dir, file in os.walk(DATA_DIR):
        if( 'tiff' in parent):
            for f in file:
                raw_stack = io.imread(os.path.join(parent, f))
                for i in range(raw_stack.shape[0]):
                    raw_img = raw_stack[i, ..., 2]
                    rb_radius = 60
                    rb_radius = 30 + np.int(i * 30 / raw_stack.shape[0])
                    file_name = os.path.join(DATA_DIR, 'mask', ('_'.join((os.path.splitext(f)[0], str(i).zfill(2))) + '.png'))
                    FindMask(file_name, raw_img, 6, [10, 2], [rb_radius, 60], .95, 2)
    t1 = time.time()
    log.write('Total time in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')
    log.close()
    # print("DONE")
