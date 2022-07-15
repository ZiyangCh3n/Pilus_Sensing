import matplotlib.pyplot as plt
from skimage import io, filters, util, restoration, morphology
from os import path, getcwd, walk, mkdir
import time
import numpy as np
import sys
import json

ARG = sys.argv
DATA_DIR = path.join(path.dirname(getcwd()), 'data', ARG[1])
T_START = time.asctime(time.localtime(time.time()))
START = int(ARG[2])
if not int(ARG[3]): # set to 0 to run through all files
    ARG[3] = 10000
STOP = int(ARG[2]) + int(ARG[3]) - 1
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}
PARAS = {
    'MASK': {'close_radius': 6, # the radius for the closing step
              'sharp_radius': 10, # the radius for the sharp step
              'sharp_amount': 2, # the amount fo r the sharp step
              'rolling_ball_radius_lower_bound': 100, # the radius for the rolling ball step
              'rolling_ball_radius_upper_bound': 200, 
              'gaussian_blur_radius': 150, # the radius for the gaussian blur
              'hysteresis_tolerance': .9, # the tolerance for hysteresis 
              'tophat_radius': 2, # the radius for the white tophat
              'histogram_nbin': 513, # number of bins for histogram
              'flat': 10, # this would make diff less than 10 as 0
              'jump_thr': 400, # the step allowed for threshold abosulute value jump
              'jump_area': .35}, # the step allowed for area proportion jump
    'MIN': {'window_width': 10, # the window width for moving average
             'flat': 1, # this would make diff 0 as 0
             'upper_bound': 4000, # the upper bound for threshold value
             'lower_bound': 600} # the lower bound for threshold value
}
HARD_THRESH = 4000 # force threshold to be this value, STUPID, set to 0 to mute

def SmoothByAvg(window_width, x, y):
    cumsum = np.cumsum(y)
    half_width = int(window_width / 2)
    moving_avg = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    moving_bins = x[half_width:-half_width]
    return moving_bins, moving_avg

def FindMin(hist, bins, thresholds, window_width, flat, ub, bb):
    hist = hist[1:]
    bins = bins[1:]
    width_half = int(window_width / 2)
    bins_s, hist_s = SmoothByAvg(window_width, bins, hist)
    diff = np.diff(hist_s, 1)
    bins_ss, diff_s = SmoothByAvg(window_width, bins_s[1:-1], diff)
    bins_sss, diff_ss = SmoothByAvg(window_width, bins_ss, diff_s)
    diff_ss = np.int16(np.abs(diff_ss) / flat) * flat
    thresh_m = bins_sss[np.argmin(diff_ss)]
    thresholds = np.sort(np.append(thresholds, thresh_m))
    thresholds = [thr for thr in thresholds if bb < thr < ub]
    if not len(thresholds):
        thresholds = [1000, 2000, 3000, 4000, 5000]
    inds = [np.where(bins_sss > thr)[0][0] for thr in thresholds]
    lb = [idx - width_half if idx > width_half else 0 for idx in inds]
    rb = [idx + width_half if idx + width_half < len(bins_sss) -1 else len(bins_sss) -1 for idx in inds]
    diff_ss_min = [np.min(diff_ss[a:b]) for a, b in zip(lb, rb)]
    bins_sss_min = bins_sss[np.sum([[np.argmin(diff_ss[a:b]) for a, b in zip(lb, rb)], lb], axis = 0)]
    area_exposed = list(map(lambda x: np.sum(hist[bins[:-1] >= x]), bins_sss_min))
    # plt.title(thresh_m)
    # plt.plot(bins_sss, diff_ss)
    # for bin, dif in zip(bins_sss_min, diff_ss_min):
    #     plt.axvline(bin, label = '%d-%d' % (bin, dif))
    #     plt.legend()
    # plt.show()
    # plt.close()
    return diff_ss_min, bins_sss_min, thresh_m, area_exposed

# @profile
def FindMask(file_name, img_raw, paras_close, paras_sharp, paras_rb, paras_gauss, paras_hys, paras_hat, paras_hist, flat, jump, jump_a):
    img_filled = morphology.closing(img_raw, morphology.disk(paras_close))
    # img_filled = morphology.diameter_closing(img_raw, diameter_threshold = 6)
    # img_filled = morphology.opening(img_filled, morphology.disk(1))
    img_sharp = filters.unsharp_mask(img_filled, 
                                        radius = paras_sharp[0], 
                                        amount = paras_sharp[1],
                                        preserve_range = True)
    bg = restoration.rolling_ball(img_sharp, radius = paras_rb)
    # bg_normal = util.img_as_int(filters.rank.mean(util.img_as_int(bg.astype(int)), selem=morphology.disk(paras_rb[1])))
    bg_normal = util.img_as_uint(filters.rank.mean(util.img_as_uint(bg.astype(int)), selem=morphology.disk(paras_gauss)))
    img_bg_reduced = img_sharp - bg_normal
    img_bg_reduced[img_sharp < bg_normal] = 0
    thresholds = filters.threshold_multiotsu(img_bg_reduced, classes = 4)
    img_otsu = np.digitize(img_bg_reduced, bins = thresholds)
    hist, bins = np.histogram(img_bg_reduced.ravel(), bins=paras_hist)
    # inds = [np.where(bins > thr)[0][0] for thr in thresholds]
    # diff = np.abs(hist[[i + paras_hist[1] for i in inds]] - hist[[i - paras_hist[1] for i in inds]])
    diff, thresh_min, thresh_m, area_exposed = FindMin(hist, bins, thresholds, window_width = PARAS_MIN['window_width'], 
                                                       flat = PARAS_MIN['flat'], ub = PARAS_MIN['upper_bound'], bb = PARAS_MIN['lower_bound'])
    idx = np.argmin([i // flat * flat for i in diff])
    thresh = thresh_min[idx]
    area = area_exposed[idx]
    # thresh = thresholds[np.argmin(diff)]
    # thresh_min_f0, thresh_min_f = [thresh], [thresh]
    if len(AREA + THRESH):
        if (area - AREA[-1]) > jump_a * AREA[-1] or np.abs(thresh - THRESH[-1]) > jump:
            inds_area = [idx for idx in range(len(area_exposed)) if 0 < (area_exposed[idx] - AREA[-1]) <= jump_a * AREA[-1]]
            inds_thr = [idx for idx in range(len(thresh_min)) if np.abs(thresh_min[idx] - THRESH[-1]) <= jump]
            u_inds = list(set(inds_area).union(set(inds_thr)))
            # thresh_min_f0 = [thresh_min[i] for i in range(len(area_exposed)) if  0 < (area_exposed[i] - AREA[-1]) <= jump_a * AREA[-1]]
            # area_f0 = [area for area in area_exposed if  0 < (area - AREA[-1]) <= jump_a * AREA[-1]]
            # thresh_min_f = [thr for thr in thresh_min if np.abs(thr - THRESH[-1]) <= jump]
            # area_f = [area_exposed[i] for i in range(len(thresh_min)) if np.abs(thresh_min[i] - THRESH[-1]) <= jump]
            # u_thr = list(set(thresh_min_f0).union(thresh_min_f))
            if len(u_inds):
                idx_thr = u_inds[np.argmin(np.abs(thresh_min[u_inds] - THRESH[-1]))]
                thresh = thresh_min[idx_thr]
                area = area_exposed[idx_thr]
            else:
                thresh = THRESH[-1]
                area = AREA[-1]
    # if len(AREA):
    #     if (area - AREA[-1]) > jump_a * AREA[-1]:
    #         thresh_min_f0 = [thresh_min[i] for i in range(len(thresh_min)) if  0 < (area - AREA[-1]) <= jump_a * AREA[-1]]
                
    # if len(THRESH):
    #     if np.abs(thresh - THRESH[-1]) > jump:
    #         thresh_min_f = [thr for thr in thresh_min if np.abs(thr - THRESH[-1]) <= jump]
            # if len(thresh_min_f):
            #     idx = np.argmin(np.abs(thresh_min_f - THRESH[-1]))
            #     thresh = thresh_min_f[idx]
            # else:
            #     thresh = THRESH[-1]
    # if len(AREA + THRESH):
    #     u = list(set(thresh_min_f0).union(thresh_min_f))
    #     if len(u):
    #         idx_thr = np.argmin(np.abs(u - THRESH[-1]))
    #         thresh = u[idx_thr]
    #     else:
    #         thresh = THRESH[-1]
    #         area = AREA[-1]
    if HARD_THRESH:
        thresh = HARD_THRESH
    THRESH.append(thresh)
    AREA.append(area)
    
    low = thresh * paras_hys
    high = thresh
    img_hyst = filters.apply_hysteresis_threshold(img_bg_reduced, low, high)
    img_hyst = morphology.binary_closing(img_hyst, selem = morphology.disk(paras_hat * 2)).astype(int)
    res = morphology.white_tophat(img_hyst, morphology.disk(paras_hat))
    img_res = img_hyst - res.astype(int)

    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (10, 10))
    ax[0, 0].imshow(img_raw, cmap = 'gray')
    ax[0, 0].set_title('Raw')
    ax[0, 1].imshow(img_sharp, cmap = 'gray')
    ax[0, 1].set_title('Sharpened')
    ax[0, 2].imshow(img_bg_reduced, cmap = 'gray')
    ax[0, 2].set_title('BG reduced, r = %d' % paras_rb)
    ax[1, 0].imshow(img_otsu, cmap = 'turbo')
    ax[1, 0].set_title('OTSU')
    ax[1, 1].imshow(img_hyst, cmap = 'gray')
    ax[1, 1].set_title('Hysteresis')
    ax[1, 2].imshow(img_res, cmap = 'gray')
    ax[1, 2].set_title('Tophat')
    ax[2, 0].hist(img_bg_reduced.ravel(), bins = paras_hist)
    for thr, dif in zip(thresh_min, diff):
        if thr == thresh:
            if thr != thresh_m:
                ax[2, 0].axvline(thr, color = 'b', label = '%d-%d' % (thr, dif))
            else:
                ax[2, 0].axvline(thr, color = 'g', label = '%d-%d' % (thr, dif))
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
    return img_res

def Reshape(raw):
    tpcount = int(raw.shape[0] / 3)
    temp = [raw[np.arange(tpcount) * 3, ...], raw[np.arange(tpcount) * 3 + 1, ...], raw[np.arange(tpcount) * 3 + 2, ...]]
    out = np.stack(temp, axis=-1)
    return out

if __name__ == '__main__':
    LOC = 0
    for parent_m, folder_m, file_m in walk(DATA_DIR):
        if 'ref.txt' in file_m:
            flag = False
            t0 = time.time()
            data_dir = parent_m
            if not path.exists(path.join(data_dir, 'paras.json')):
                with open(path.join(data_dir, 'paras.json'), 'w') as file:
                    js = json.dump(PARAS, file)
            with open(path.join(data_dir, 'paras.json'), 'r') as file:
                paras = json.load(file)
            PARAS_MASK = paras['MASK']
            PARAS_MIN = paras['MIN']
            rb_range = PARAS_MASK['rolling_ball_radius_upper_bound'] - PARAS_MASK['rolling_ball_radius_lower_bound']
            if not path.exists(path.join(data_dir, 'mask_ref')):
                mkdir(path.join(data_dir, 'mask_ref'))
            if not path.exists(path.join(data_dir, 'mask')):
                mkdir(path.join(data_dir, 'mask'))
            for parent, dir, file in walk(data_dir):
                if( 'tiff' in parent):
                    file.sort()
                    for f in file:
                        if START <= LOC <= STOP:
                            flag = True
                            raw_stack = io.imread(path.join(parent, f))
                            if len(raw_stack.shape) < 4:
                                raw_stack = Reshape(raw_stack)
                            THRESH = []
                            AREA = []
                            for i in range(raw_stack.shape[0]):
                                raw_img = raw_stack[i, ..., CHANNEL['mcherry']]
                                rb_radius = PARAS_MASK['rolling_ball_radius_lower_bound'] + int(i * rb_range / raw_stack.shape[0])
                                gauss_radius = int(rb_radius * 1.5)
                                file_name = path.join(data_dir, 'mask_ref', ('_'.join((path.splitext(f)[0], str(i).zfill(2))) + '.png'))
                                if path.exists(file_name):
                                    continue
                                mask = FindMask(file_name, raw_img,  paras_close = PARAS_MASK['close_radius'],
                                                paras_sharp = [PARAS_MASK['sharp_radius'], PARAS_MASK['sharp_amount']],  
                                                paras_rb = rb_radius, paras_gauss = gauss_radius, paras_hys = PARAS_MASK['hysteresis_tolerance'], 
                                                paras_hat = PARAS_MASK['tophat_radius'], paras_hist = PARAS_MASK['histogram_nbin'], 
                                                flat = PARAS_MASK['flat'], jump = PARAS_MASK['jump_thr'], jump_a = PARAS_MASK['jump_area'])
                                # np.save(file_name.replace('mask_ref', 'mask', 1).replace('.png', '.npy', 1), np.bool_(mask))
                                io.imsave(file_name.replace('mask_ref', 'mask', 1), mask)
                                # np.savetxt(file_name.replace('mask_ref', 'mask', 1).replace('.png', '.csv', 1), mask, delimiter = ',')
                        LOC += 1
            t1 = time.time()
            if flag:
                with open(path.join(data_dir, 'log'), 'a') as log:
                    log = open(path.join(data_dir, 'log'), 'a')
                    log.write('-'*10 + 'Masking' + '-'*10 + '\n')
                    log.write('Started at: ' + T_START + '\n')
                    log.write('DATA_DIR: ' + data_dir + '\n')
                    log.write('START: %d STOP: %d \n' % (START, STOP))
                    log.write('Total time in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')
