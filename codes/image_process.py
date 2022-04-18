import numpy as np
import pandas as pd
# from nd2reader import ND2Reader
# import matplotlib.pyplot as plt
# from skimage import io
from os import path, getcwd, walk, makedirs, rename
from subprocess import run
import sys
import time

ARGS = sys.argv
DATA_DIR = path.join(path.dirname(getcwd()), 'data', ARGS[1])
T_START = time.asctime(time.localtime(time.time()))
CHANNEL = {'phase': 0, 'mcherry': 2, 'YFP': 1}
BFTOOL_DIR = path.join(path.dirname(getcwd()), 'bftools')
skip_preprocess = False

def Rename_Files(data_dir):
    ref = {}
    for line in open(path.join(data_dir, 'ref.txt')):
        a = line.strip().split(',')
        ref[a[0]] = a[1:]

    for parent, dir, file in walk(data_dir):
        if not (len(dir) or 'tiff' in parent):
            n_replicate = int(len(file) / len(ref['conc']))
            rename_list = ['_'.join([strain, str(float(conc)), str(rep).zfill(2)]) + '.nd2' 
            for strain in ref['strain'] 
            for conc in ref['conc']
            for rep in np.arange(n_replicate)]
            print(rename_list)
            for f,r in zip(file, rename_list):
                if path.exists(path.join(parent, r)):
                    continue
                rename(path.join(parent, f), path.join(parent, r))
            # if not('_nd2' in parent):    
            #     os.rename(parent, parent + '_nd2')
            make_folder = parent + '_tiff'
            if not path.exists(make_folder):
                makedirs(make_folder)

def GetTiff(data_dir):
    # subprocess.run(['chmod', '+x', './bfconvert'], cwd = BFTOOL_DIR, shell = True)
    # subprocess.run([BFTOOL_DIR, 'chmod', '+x', './bfconvert'])
    for parent, dir, file in walk(data_dir):
        file = [f for f in file if f.endswith('.nd2')] 
        file.sort()
        if len(file):
            print(file)
            for f ,i in zip(file, np.arange(len(file))):
                input_dir = path.join(parent, f)
                output_dir = path.join(parent + '_tiff', f.replace('.nd2', '.tiff'))
                if path.exists(output_dir):
                    print("EXIST: %s" % output_dir)
                    continue
                # command = 'bfconvert ' + input_dir + ' ' + output_dir
                # subprocess.run(['./bfconvert', input_dir, output_dir], cwd = BFTOOL_DIR, shell = True)
                # subprocess.Popen(['./bfconvert', input_dir, output_dir], cwd = BFTOOL_DIR, shell = True) # This is faster
                run([path.join(BFTOOL_DIR, 'bfconvert'), input_dir, output_dir], shell = sys.platform == 'win32')
                # print(f'{i + 1} out of {len(file)} convertion finished.')

if __name__ == '__main__':
    if not skip_preprocess:
        for parent, dir, file in walk(DATA_DIR):
            # if not (len(dir) or 'tiff' in parent):
            if all(['.nd2' in f for f in file]) and len(file):
                print(parent)
                t0 = time.time()
                data_dir = path.dirname(parent)
                Rename_Files(data_dir)
                GetTiff(data_dir)
                t1 = time.time()
                with open(path.join(data_dir, 'log'), 'a') as log:
                    log.write('-' * 10 + 'CONVERSION' + '-' * 10 + '\n')
                    log.write('Created at: ' + T_START + '\n')
                    log.write('DATA_DIR: ' + data_dir + '\n')
                    log.write('Total time in min: ' + str(round((t1 - t0) /60, 2)) + '\n')
                

    # log.close()  

    # dir_list = {}
    # for parent, dir, file in os.walk(DATA_DIR):
    #     if not len(dir):
    #         bsname = os.path.basename(parent).split('_')[-1]
    #         dir_list[bsname] = [os.path.join(parent, f) for f in file]


    # for tiff_dir, nd2_dir in zip(dir_list['tiff'], dir_list['nd2']):
    #     nd2_img = ND2Reader(nd2_dir)
    #     tiff_stack = io.imread(tiff_dir)
    #     channels = nd2_img.metadata['channels']
    #     timestep = np.round(nd2_img.timesteps / 1000 / 60, 0)
    #     print(tiff_dir)
    #     print(timestep)
    #     n_timepoint = tiff_stack.shape[0]
# for img, nd2 in zip(img_dir, nd2_dir):
#     nd2_img = ND2Reader(nd2)
#     img_stack = io.imread(img)
#     channels = nd2_img.metadata['channels']
#     timestep = np.round(nd2_img.timesteps / 1000 / 60, 0)
#     n_position = img_stack.shape[0]
#     n_timepoint = img_stack.shape[1] 



