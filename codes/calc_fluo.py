import matplotlib.pyplot as plt
import numpy as np
from skimage import io, restoration, filters, util, morphology
import sys
from os import path, walk, mkdir, getcwd, listdir
import time
import pandas as pd

ARG = sys.argv
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}
DATA_DIR = path.join(path.dirname(getcwd()), 'data', ARG[1])
T_START = time.asctime(time.localtime(time.time()))
START = int(ARG[2])
if not int(ARG[3]): # set to 0 to run through all files
    ARG[3] = 10000
STOP = int(ARG[2]) + int(ARG[3]) - 1


def GetMasked(img, mask, filename, channel, paras_rb = [150, 225]):
    img_raw = img[..., CHANNEL[channel]]
    bg = restoration.rolling_ball(img_raw, radius = paras_rb[0])
    bg_normal = util.img_as_uint(filters.rank.mean(util.img_as_uint(bg.astype(int)), selem = morphology.disk(paras_rb[1])))
    img_bg_reduced = img_raw - bg_normal
    img_bg_reduced[img_raw < bg_normal] = 0
    img_masked = img_bg_reduced * mask

    x = [img_raw.ravel(), bg_normal.ravel(), img_bg_reduced.ravel()]
    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    ax[0, 0].imshow(img_raw, cmap = 'gray')
    ax[0, 0].set_title('%s channel raw' % channel)
    ax[0, 1].imshow(mask, cmap = 'gray')
    ax[0, 1].set_title('Mask')
    ax[0, 2].imshow(img_bg_reduced, cmap = 'gray')
    ax[0, 2].set_title('BG reduced')
    ax[1, 0].imshow(img_masked, cmap = 'gray')
    ax[1, 0].set_title('Masked')
    ax[1, 1].imshow(bg_normal, cmap = 'gray')
    ax[1, 1].set_title('Background') 
    ax[1, 2].hist(np.array(x).T, 200, density = True, histtype = 'step', stacked = False, fill = False, label = ['raw', 'bg', 'bg_reduced'])
    ax[1, 2].legend(loc = 'upper left')
    ax[1, 2].set_title('Hitogram - gray value')
    for a in ax.ravel()[:-1]:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(path.join(data_dir, (channel + '_ref'), filename), bbox_inches = 'tight')
    # plt.show()
    plt.close()
    return img_masked, bg_normal * mask

# @profile
def CalcFluo(img_tiff, dir_mask, filename, timepoint):
    # img_stack = io.imread(dir_img)
    # img_raw = img_stack[timepoint, ..., YFP]
    mask = np.bool_(io.imread(dir_mask)) * 1
    yfp_masked, yfp_bg = GetMasked(img_tiff, mask, filename, 'YFP')
    mcherry_masked, mcherry_bg = GetMasked(img_tiff, mask, filename, 'mcherry')
    returnval = np.sum([mask, yfp_masked, mcherry_masked, yfp_bg, mcherry_bg], axis = (1, 2), dtype = np.uint32)
    return returnval

if __name__ == '__main__':
    LOC = 0
    for parent_m, folder_m, file_m in walk(DATA_DIR):
        if 'ref.txt' in file_m:
            flag = False
            t0 = time.time()
            data_dir = parent_m
            img_dir = path.join(data_dir, [folder for folder in listdir(data_dir) if folder.endswith('tiff')][0])
            mask_dir = path.join(data_dir, 'mask')
            if not path.exists(path.join(data_dir, 'YFP_ref')):
                mkdir(path.join(data_dir, 'YFP_ref'))
            if not path.exists(path.join(data_dir, 'mcherry_ref')):
                mkdir(path.join(data_dir, 'mcherry_ref'))
            for parent_img, dir_img, file_img in walk(img_dir):
                file_img.sort()
                for f_img in file_img:
                    if START <= LOC <= STOP:
                        flag = True
                        dir_img = path.join(parent_img, f_img)
                        img_tiff = io.imread(dir_img)
                        pd_dict = {'Label': [], 
                        'Area': [], 
                        'YFP intensity total': [], 
                        'YFP background': [],
                        'mCherry intensity total': [],
                        'mCherry background': [], 
                        'Time': []}
                        for parent, dir, file in walk(mask_dir):
                            file.sort()
                            for f in file:
                                if f_img.removesuffix('.tiff').split('_')[2] == f.removesuffix('.png').split('_')[2]:
                                    if path.exists(path.join(data_dir, 'YFP_ref', f)):
                                        continue
                                    dir_mask = path.join(parent, f)
                                    timepoint = int(f.removesuffix('.png').split('_')[3])
                                    vals = CalcFluo(img_tiff[timepoint, ...], dir_mask, f, timepoint)
                                    pd_dict['Area'].append(vals[0])
                                    pd_dict['YFP intensity total'].append(vals[1])
                                    pd_dict['YFP background'].append(vals[2])
                                    pd_dict['mCherry intensity total'].append(vals[3])
                                    pd_dict['mCherry background'].append(vals[4])
                                    pd_dict['Time'].append(timepoint)
                                    pd_dict['Label'].append(f_img)
                        if len(pd_dict['Area']):
                            df = pd.DataFrame(pd_dict, dtype = 'uint32')
                            df.to_csv(path.join(data_dir, 'fluorescence.csv'), index=False, mode='a', 
                            header=not path.exists(path.join(data_dir, 'fluorescence.csv')))
                    LOC += 1
            t1 = time.time()
            if flag:
                with open(path.join(data_dir, 'log'), 'a') as log:
                    log.write('-'*10 + 'Fluorescence' + '-'*10 + '\n')
                    log.write('Started at: ' + T_START + '\n')
                    log.write('DATA_DIR: ' + data_dir + '\n')
                    log.write('START: %d STOP: %d' % (START % 36, STOP % 36))
                    log.write('Totaltime in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')

