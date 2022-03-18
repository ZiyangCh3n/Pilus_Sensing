import numpy as np
import pandas as pd
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
from skimage import io
import os
import subprocess

WORK_DIR = r'D:\Researchdata\031722'
BFTOOL_DIR = r'C:\Users\littl\Documents\Research\pilus\bftools'
skip_preprocess = True

def Rename_Files():
    ref = {}
    for line in open(os.path.join(WORK_DIR, 'ref.txt')):
        a = line.strip().split(',')
        ref[a[0]] = a[1:]

    for parent, dir, file in os.walk(WORK_DIR):
        if not len(dir):
            n_replicate = int(len(file) / len(ref['conc']))
            rename_list = ['_'.join([strain, conc, str(rep)]) + '.nd2' 
            for strain in ref['strain'] 
            for conc in ref['conc']
            for rep in np.arange(n_replicate)]
            print(rename_list)
            for f,r in zip(file, rename_list):
                os.rename(os.path.join(parent, f), os.path.join(parent, r))
            os.rename(parent, parent + '_nd2')
            make_folder = parent + '_tiff'
            if not os.path.exists(make_folder):
                os.makedirs(make_folder)

def GetTiff():
    for parent, dir, file in os.walk(WORK_DIR):
        file = [f for f in file if f.endswith('.nd2')] 
        if len(file):
            print(file)
            for f ,i in zip(file, np.arange(len(file))):
                input_dir = os.path.join(parent, f)
                output_dir = os.path.join(os.path.join(parent.replace('_nd2', '_tiff'), f.replace('.nd2', '.tiff')))
                # command = 'bfconvert ' + input_dir + ' ' + output_dir
                # subprocess.run(['bfconvert', input_dir, output_dir], cwd = BFTOOL_DIR, shell = True)
                subprocess.Popen(['bfconvert', input_dir, output_dir], cwd = BFTOOL_DIR, shell = True) # This is faster
                print(f'{i + 1} out of {len(file)} convertion finished.')

if __name__ == '__main__':
    if not skip_preprocess:
        Rename_Files()
        GetTiff()  

    dir_list = {}
    for parent, dir, file in os.walk(WORK_DIR):
        if not len(dir):
            bsname = os.path.basename(parent).split('_')[-1]
            dir_list[bsname] = [os.path.join(parent, f) for f in file]


    for tiff_dir, nd2_dir in zip(dir_list['tiff'], dir_list['nd2']):
        nd2_img = ND2Reader(nd2_dir)
        tiff_stack = io.imread(tiff_dir)
        channels = nd2_img.metadata['channels']
        timestep = np.round(nd2_img.timesteps / 1000 / 60, 0)
        print(tiff_dir)
        print(timestep)
        n_timepoint = tiff_stack.shape[0]
# for img, nd2 in zip(img_dir, nd2_dir):
#     nd2_img = ND2Reader(nd2)
#     img_stack = io.imread(img)
#     channels = nd2_img.metadata['channels']
#     timestep = np.round(nd2_img.timesteps / 1000 / 60, 0)
#     n_position = img_stack.shape[0]
#     n_timepoint = img_stack.shape[1] 


print('done')


