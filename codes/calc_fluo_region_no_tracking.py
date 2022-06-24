tracking = False
def CalcFluoMain(img, mask_dir, lb):
    centroid0 = []
    ids0 = []
    # dict_df = {}
    
    for parent, folder, file in walk(mask_dir):
        file.sort()
        for f in file:
            label_mask = f.removesuffix('.png')[:-3]
            tp_mask = int(f.removesuffix('.png').split('_')[3])
            pos_mask = int(f.removesuffix('.png').split('_')[2])
            if label_mask == lb:
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
        df.to_csv(path.join(data_dir, 'fluorescence',  str(START) + '-' + str(STOP) + '.csv'), index = False, mode = 'a',
              header = not path.exists(path.join(data_dir, 'fluorescence',  str(START) + '-' + str(STOP) + '.csv')))
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
            if not path.exists(path.join(data_dir, 'fluorescence')):
                mkdir(path.join(data_dir, 'fluorescence'))
            for parent_img, folder_img, file_img in walk(imgfolder_dir):
                file_img.sort()
                for f in file_img:
                    if START <= LOC <= STOP:
                        flag = True
                        img_dir = path.join(parent_img, f)
                        img = io.imread(img_dir)
                        # pos = int(f.removesuffix('.tiff').split('_')[2])
                        lb = f.removesuffix('.tiff')
                        CalcFluoMain(img, maskfolder_dir, lb)
                    LOC += 1
            t1 = time.time()
            if flag:
                with open(path.join(data_dir, 'log'), 'a') as log:
                    log.write('-'*10 + 'Fluorescence' + '-'*10 + '\n')
                    log.write('Started at: ' + T_START + '\n')
                    log.write('DATA_DIR: ' + data_dir + '\n')
                    log.write('START: %d STOP: %d \n' % (START % 36, STOP % 36))
                    log.write('Totaltime in min: ' + str(round((t1 - t0) / 60, 2)) + '\n')