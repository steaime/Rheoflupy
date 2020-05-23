import sys
import os
import numpy as np
import configparser
import warnings
import time
import openpiv
from openpiv import windef
from pims import ImageSequence
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

if os.name == 'nt':
    path_sep = '\\'
else:
    path_sep = '/'


ProgramRoot = os.path.dirname(sys.argv[0]) + path_sep
sys.path.append(ProgramRoot)
import BinaryImgs
import SharedFunctions

def ReadPIVsettings(config):
    settings = windef.Settings()
    settings.filepath_images             = SharedFunctions.ConfigGet(config, 'input',      'input_folder')
    settings.frame_pattern_a             = SharedFunctions.ConfigGet(config, 'input',      'filter_frameA', '', str)
    settings.frame_pattern_b             = SharedFunctions.ConfigGet(config, 'input',      'filter_frameB', '', str)
    settings.ROI                         = SharedFunctions.ConfigGet(config, 'input',      'ROI')
    settings.dynamic_masking_method      = 'None'
    settings.dynamic_masking_threshold   = 0.005
    settings.dynamic_masking_filter_size = 7
    settings.correlation_method          = SharedFunctions.ConfigGet(config, 'processing', 'correlation_method')
    settings.iterations                  = SharedFunctions.ConfigGet(config, 'processing', 'iterations', 1, int)
    settings.windowsizes                 = SharedFunctions.ConfigGet(config, 'processing', 'window_sizes', [4], int)
    settings.overlap                     = SharedFunctions.ConfigGet(config, 'processing', 'overlap', [2], int)
    settings.subpixel_method             = SharedFunctions.ConfigGet(config, 'processing', 'subpixel_method')
    settings.interpolation_order         = SharedFunctions.ConfigGet(config, 'processing', 'interpolation_order', 3, int)
    settings.scaling_factor              = SharedFunctions.ConfigGet(config, 'processing', 'pixel_size', 1.0, float)
    settings.dt                          = SharedFunctions.ConfigGet(config, 'processing', 'dt', 1.0, float)
    settings.extract_sig2noise           = SharedFunctions.ConfigGet(config, 'processing', 'extract_sig2noise', False, bool)
    settings.sig2noise_method            = SharedFunctions.ConfigGet(config, 'processing', 'sig2noise_method')
    settings.sig2noise_mask              = SharedFunctions.ConfigGet(config, 'processing', 'sig2noise_mask', 2, int)
    settings.validation_first_pass       = SharedFunctions.ConfigGet(config, 'validation', 'validation_first_pass', False, bool)
    settings.MinMax_U_disp               = SharedFunctions.ConfigGet(config, 'validation', 'MinMax_U_disp', [-1000.0, 1000.0], float)
    settings.MinMax_V_disp               = SharedFunctions.ConfigGet(config, 'validation', 'MinMax_V_disp', [-1000.0, 1000.0], float)
    settings.std_threshold               = SharedFunctions.ConfigGet(config, 'validation', 'std_threshold', 7.0, float)
    settings.median_threshold            = SharedFunctions.ConfigGet(config, 'validation', 'median_threshold', 3.0, float)
    settings.median_size                 = SharedFunctions.ConfigGet(config, 'validation', 'median_size', 1, int)
    settings.do_sig2noise_validation     = SharedFunctions.ConfigGet(config, 'validation', 'do_sig2noise_validation', False, bool)
    settings.sig2noise_threshold         = SharedFunctions.ConfigGet(config, 'validation', 'sig2noise_threshold', 1.2, float)
    settings.replace_vectors             = SharedFunctions.ConfigGet(config, 'validation', 'replace_vectors', True, bool)
    settings.smoothn                     = SharedFunctions.ConfigGet(config, 'validation', 'smoothn', True, bool)
    settings.smoothn_p                   = SharedFunctions.ConfigGet(config, 'validation', 'smoothn_p', 0.5, float)
    settings.filter_method               = SharedFunctions.ConfigGet(config, 'validation', 'filter_method')
    settings.max_filter_iteration        = SharedFunctions.ConfigGet(config, 'validation', 'max_filter_iteration', 4, int)
    settings.filter_kernel_size          = SharedFunctions.ConfigGet(config, 'validation', 'filter_kernel_size', 2.0, float)
    settings.save_path                   = SharedFunctions.ConfigGet(config, 'output',     'output_folder')
    settings.save_folder_suffix          = SharedFunctions.ConfigGet(config, 'output',     'save_folder_suffix')
    settings.save_plot                   = SharedFunctions.ConfigGet(config, 'output',     'save_plot', False, bool)
    settings.show_plot                   = SharedFunctions.ConfigGet(config, 'output',     'show_plot', False, bool)
    settings.scale_plot                  = SharedFunctions.ConfigGet(config, 'output',     'scale_plot', 1.0, float)
    return settings






if __name__ == '__main__':
    
    cmd_list = []
    
    inp_fnames = [ProgramRoot + 'PIVdef.ini']
    for argidx in range(1, len(sys.argv)):
        # If it's something like -cmd, add it to the command list
        # Otherwise, assume it's the path of some input file to be read
        if (sys.argv[argidx][0] == '-'):
            cmd_list.append(sys.argv[argidx])
        else:
            inp_fnames.append(sys.argv[argidx])

    # Read input file for configuration
    config = configparser.ConfigParser(allow_no_value=True)
    for conf_f in inp_fnames:
        print('Reading config file: ' + str(conf_f))
        config.read(conf_f)
    
    out_froot = SharedFunctions.ConfigGet(config, 'output', 'output_folder') + path_sep
    time_prefix = SharedFunctions.ConfigGet(config, 'output', 'time_prefix', '', str)
    lag_prefix = SharedFunctions.ConfigGet(config, 'output', 'lag_prefix', '', str)
    img_ext = SharedFunctions.ConfigGet(config, 'input', 'image_ext', '', str)
    settings = ReadPIVsettings(config)
    froot = settings.filepath_images
    lagtimes = SharedFunctions.ConfigGet(config, 'input', 'lagtimes', None, int)
    zrange = SharedFunctions.ConfigGet(config, 'input', 'zrange', None, int)
    zidx_len = SharedFunctions.ConfigGet(config, 'input', 'zidx_len', 4, int)
    zprefix = SharedFunctions.ConfigGet(config, 'input', 'zprefix', 'Z', str)
    aggr_subfolder = SharedFunctions.ConfigGet(config, 'output', 'aggregated_subfolder', 'Aggregated', str)
    aggr_root = out_froot + aggr_subfolder + path_sep
    vel_prefix = SharedFunctions.ConfigGet(config, 'postprocess', 'vel_prefix', '_v', str)
    refinedv_prefix = SharedFunctions.ConfigGet(config, 'postprocess', 'refinedv_prefix', '__v', str)
    plots_subfolder = SharedFunctions.ConfigGet(config, 'output', 'plots_subfolder', None, str)
    if (plots_subfolder is None):
        plot_root = None
    else:
        plot_root = out_froot + plots_subfolder + path_sep
        
    
    # If -aggregate is in the command list,
    # only aggregate results, don't go through the PIV computation
    if ('-skipPIV' in cmd_list):
        
        print('\nSkipping PIV step.')
    
    else:
        
        imgs_list = []
        if (lagtimes is None and settings.frame_pattern_b == ''):
            lagtimes = [1]
        if (lagtimes is None):
            fnamelist_A = SharedFunctions.FindFileNames(froot, Ext=img_ext, FilterString=settings.frame_pattern_a, AppendFolder=False)
            fnamelist_B = SharedFunctions.FindFileNames(froot, Ext=img_ext, FilterString=settings.frame_pattern_b, AppendFolder=False)
            for i in range(min(len(fnamelist_A), len(fnamelist_B))):
                imgs_list.append({'imgA':fnamelist_A[i],'imgB':fnamelist_B[i],'t':i,'dt':0})
            print('Processing {0} image couples: {1}*{2}*{4} - {1}*{3}*{4}'.format(len(imgs_list), froot, settings.frame_pattern_a,\
                                                                                  settings.frame_pattern_b, img_ext))
        else:
            if (zrange is None):
                z_list = ['']
            else:
                z_list = []
                if (len(zrange)<3):
                    zrange.append(1)
                for zidx in range(zrange[0], zrange[1], zrange[2]):
                    z_list.append(zprefix+str(zidx).zfill(zidx_len))
            for z_suffix in z_list:
                fnamelist = SharedFunctions.FindFileNames(froot, Ext=z_suffix+img_ext, FilterString=settings.frame_pattern_a, AppendFolder=False)
                for i in range(len(fnamelist)):
                    for dt in lagtimes:
                        if (i+dt < len(fnamelist)):
                            imgs_list.append({'imgA':fnamelist[i],'imgB':fnamelist[i+dt],'t':i,'dt':dt, 'z_suf':z_suffix})

            print('{0} image couples to be processed: {1} images in {2}*{3}*{4}, {5} lagtimes {6}'.format(len(imgs_list),\
                  len(fnamelist), froot, settings.frame_pattern_a, z_suffix+img_ext, len(lagtimes), lagtimes))
        
        str_method = SharedFunctions.ConfigGet(config, 'processing', 'PIV_method', 'standard', str)
        if (str_method == 'standard'):
            print('Using standard PIV method')
        elif (str_method == 'WiDIM'):
            raise ValueError('WiDIM method not implemented yet')
        elif (str_method == 'extended'):
            raise ValueError('extended_search_area method not implemented yet')
        else:
            raise ValueError('Method ' + str_method + ' not recognized')
        
        time_start = time.time()
        for i in range(len(imgs_list)):
            if (str_method == 'standard'):
                settings.frame_pattern_a = imgs_list[i]['imgA']
                settings.frame_pattern_b = imgs_list[i]['imgB']
                cur_outfolder = out_froot + time_prefix + str(imgs_list[i]['t']).zfill(4) + path_sep
                SharedFunctions.CheckCreateFolder(cur_outfolder)
                settings.save_path = cur_outfolder
                settings.save_folder_suffix = lag_prefix + str(imgs_list[i]['dt']).zfill(4)
                if (imgs_list[i]['z_suf'] != ''):
                    settings.save_folder_suffix = settings.save_folder_suffix + imgs_list[i]['z_suf']
                settings.show_plot = False
                
                time_step = time.time()
                windef.piv(settings)
                time_end = time.time()
                print('[{0}/{1}] Img #{2}, lag {3} processed in {4:.1f} s. Total elapsed: {5:.1f} s'.format(i+1, len(imgs_list), imgs_list[i]['t'],\
                      imgs_list[i]['dt'], time_end-time_step, time_end-time_start))
            elif (str_method == 'WiDIM'):
                # Warning: implementing this feature...
                cur_frameA = openpiv.tools.imread(froot+imgs_list[i]['imgA'])
                cur_frameB = openpiv.tools.imread(froot+imgs_list[i]['imgB'])
                if (settings.dynamic_masking_method != 'None'):
                    raise ValueError('settings.dynamic_masking_method must be None for the moment, not ' + str(settings.dynamic_masking_method))
                mark = np.ones(cur_frameA.shape, dtype=np.int32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x,y,u,v,mask=openpiv.process.WiDIM(cur_frameA.astype(np.int32), cur_frameB.astype(np.int32), mark,\
                                                       min_window_size=16, overlap_ratio=0.0, coarse_factor=2, dt=1, validation_method='mean_velocity',\
                                                       trust_1st_iter=1, validation_iter=1, tolerance=0.7, nb_iter_max=3, sig2noise_method='peak2peak')
        time_end = time.time()
        print('\n\n-----------\nDONE! {0} image pairs processed in {1:.1f} s.'.format(len(imgs_list), time_end-time_start))
    
    
    if ('-skipaggregate' in cmd_list):
        
        print('\nSkipping result aggregation')    

    elif (SharedFunctions.CheckFolderExists(out_froot) == False):
        
        print('\nNo results to aggregate! Skipping aggregation step')

    else:

        print('\nNow aggregating output results...')
        
        dirlist = SharedFunctions.FindSubfolders(out_froot, FirstLevelOnly=True, Prefix=time_prefix, Sort='ASC')
        print('{0} "{1}{2}*{3}" subfolders found'.format(len(dirlist), out_froot, time_prefix, path_sep))
        
        laglist = SharedFunctions.ExtractIndexFromStrings(SharedFunctions.FindSubfolders(dirlist[0], FirstLevelOnly=True,\
                                     Prefix='Open_PIV_results_', FilterString=lag_prefix), index_pos=-1, index_notfound=-1)
        print('{0} times found, {1} lags found in first folder'.format(len(dirlist), len(laglist)))
        
        if (SharedFunctions.CheckFolderExists(aggr_root)):
            if ('-silent' in cmd_list):
                print('Folder {0} already present. Results will be overwritten'.format(aggr_root))
            else:
                if (SharedFunctions.query_yes_no('Folder {0} already present. Rename it?'.format(aggr_root), default="yes")):
                    sub_idx = 1
                    while SharedFunctions.CheckFolderExists(out_froot + 'Aggregated_' + str(sub_idx) + path_sep):
                        sub_idx += 1
                    aggr_root_renamed = out_froot + 'Aggregated_' + str(sub_idx) + path_sep
                    SharedFunctions.RenameDirectory(out_froot + aggr_subfolder, out_froot + 'Aggregated_' + str(sub_idx))
                    print('Folder originally named {0} renamed to {0}'.format(aggr_root, aggr_root_renamed))
                else:
                    print('Results will be overwritten')                
        SharedFunctions.CheckCreateFolder(aggr_root)
        
        # Find coordinates
        img_list = SharedFunctions.FindFileNames(settings.filepath_images, Ext=SharedFunctions.ConfigGet(config, 'input', 'image_ext', '', str),\
                                                      FilterString=settings.frame_pattern_a, AppendFolder=False)
        frame_test = openpiv.tools.imread(settings.filepath_images + img_list[0])
        coords = openpiv.process.get_coordinates(image_size=frame_test.shape, window_size=settings.windowsizes[settings.iterations-1],\
                                        overlap=settings.overlap[settings.iterations-1])
        print('Image shape: {0}. Window size: {1}. Overlap: {2}. PIV grid shape: {3}x{4}'.format(frame_test.shape, settings.windowsizes[settings.iterations-1],\
                                                                  settings.overlap[settings.iterations-1], len(coords[0]), len(coords[1])))
        
        # Binary files for aggregated data
        strext = '_' + str(len(coords[0])).zfill(4) + 'x' + str(len(coords[1])).zfill(4) + 'x' + str(len(dirlist)).zfill(4) + '.raw'
        displ_x = []
        displ_y = []
        mask = []
        signoise = []
        for lidx in range(len(laglist)):
            cur_suffix = lag_prefix + str(laglist[lidx]).zfill(4) + strext
            displ_x.append(BinaryImgs.OpenMIfileForWriting(aggr_root + 'dx_' + cur_suffix))
            displ_y.append(BinaryImgs.OpenMIfileForWriting(aggr_root + 'dy_' + cur_suffix))
            mask.append(BinaryImgs.OpenMIfileForWriting(aggr_root + 'mask_' + cur_suffix))
            signoise.append(BinaryImgs.OpenMIfileForWriting(aggr_root + 'sn_' + cur_suffix))
        
        time_start = time.time()
        for tidx in range(len(dirlist)):
            time_step = time.time()
            cur_t = SharedFunctions.LastIntInStr(dirlist[tidx])
            subdirlist = SharedFunctions.FindSubfolders(dirlist[tidx], FirstLevelOnly=True, Prefix='Open_PIV_results_',\
                                                        FilterString=SharedFunctions.ConfigGet(config, 'output', 'lag_prefix', '', str))
            for lidx in range(len(laglist)):
                piv_data = None
                if (lidx < len(subdirlist)):
                    if (SharedFunctions.CheckFileExists(subdirlist[lidx]+path_sep+'field_A000.txt')):
                        piv_data = np.loadtxt(subdirlist[lidx]+path_sep+'field_A000.txt', dtype=float, comments='#', usecols=(2,3,4,5))
                if (piv_data is None):
                    piv_data = np.ones((len(coords[0])*len(coords[1]), 4), dtype=float)*np.nan
                BinaryImgs.AppendToMIfile(displ_x[lidx], piv_data[:,0], 'f')
                BinaryImgs.AppendToMIfile(displ_y[lidx], piv_data[:,1], 'f')
                BinaryImgs.AppendToMIfile(signoise[lidx], piv_data[:,2], 'f')
                BinaryImgs.AppendToMIfile(mask[lidx], piv_data[:,3], 'B')
            
            time_end = time.time()
            print('[{0}/{1}] Data for time {2} aggregated in {3:.1f} s. Elapsed time: {4:.1f} s'.format(tidx+1, len(dirlist),\
                                                                          cur_t, time_end-time_step, time_end-time_start))
            
        for lidx in range(len(laglist)):
            displ_x[lidx].close()
            displ_y[lidx].close()
            mask[lidx].close()
            signoise[lidx].close()

    if ('-skippostproc' in cmd_list):

        print('\nSkipping post processing')
    
    else:
        
        print('\nNow postprocessing results to extract velocities...')
        
        vel_method = SharedFunctions.ConfigGet(config, 'postprocess', 'vel_method', 'linreg', str)
        save_vsq = SharedFunctions.ConfigGet(config, 'postprocess', 'save_vsq', False, bool)
        if (vel_method == 'none'):
            
            raise ValueError('Postprocessing method ' + vel_method + ' not implemented yet')
            
        if (vel_method == 'linreg'):
            max_lag_list = SharedFunctions.ConfigGet(config, 'postprocess', 'max_lag', -1, int)
            
            res_filenames = [SharedFunctions.FindFileNames(aggr_root, Prefix='dx_'+lag_prefix, Ext='.raw'),\
                            SharedFunctions.FindFileNames(aggr_root, Prefix='dy_'+lag_prefix, Ext='.raw'),\
                            SharedFunctions.FindFileNames(aggr_root, Prefix='mask_'+lag_prefix, Ext='.raw'),\
                            SharedFunctions.FindFileNames(aggr_root, Prefix='sn_'+lag_prefix, Ext='.raw')]
            imgInfo = [BinaryImgs.MIinfoFromName(res_filenames[0][0], byteFormat='f'),\
                       BinaryImgs.MIinfoFromName(res_filenames[1][0], byteFormat='f'),\
                       BinaryImgs.MIinfoFromName(res_filenames[2][0], byteFormat='B'),\
                       BinaryImgs.MIinfoFromName(res_filenames[3][0], byteFormat='f')]
            for cur_info in imgInfo:
                cur_info['hdr_size'] = 0
            
            res_files = []
            for fidx in range(len(res_filenames)):
                res_files.append([])
                for lidx in range(len(laglist)):
                    res_files[fidx].append(BinaryImgs.LoadMIfile(aggr_root + res_filenames[fidx][lidx], MI_info=imgInfo[fidx].copy(), returnHeader=False))
            
            laglist = SharedFunctions.ExtractIndexFromStrings(res_filenames[0], index_pos=0, index_notfound=0)
            
            fout_list = []
            lag_selidx_list = []
            lag_selval_list = []
            max_laglen_idx = 0
            for midx in range(len(max_lag_list)):
                
                strext = '_' + '_' + lag_prefix + str(max_lag_list[midx]) + '_' + str(imgInfo[0]['img_width']).zfill(4) +\
                                   'x' + str(imgInfo[0]['img_height']).zfill(4) + 'x' + str(imgInfo[0]['img_num']).zfill(4) + '.raw'
                
                do_add = True
                if (SharedFunctions.CheckFileExists(aggr_root + vel_prefix + 'x' + strext) and SharedFunctions.CheckFileExists(aggr_root + vel_prefix + 'y' + strext)):
                    overwrite_existing = SharedFunctions.ConfigGet(config, 'postprocess', 'overwrite_existing', 'ask', str)
                    if (overwrite_existing == 'no'):
                        do_add = False
                    elif (overwrite_existing == 'ask' and not SharedFunctions.query_yes_no('  For lagtime #{0} ({1}), output files are already in folder {2}. Overwrite them?', default="no")):
                        do_add = False
                
                if (do_add):
                    lag_selidx_list.append([])
                    lag_selval_list.append([])
                    for lidx in range(len(laglist)):
                        if (max_lag_list[midx] < 0 or laglist[lidx] <= max_lag_list[midx]):
                            lag_selidx_list[midx].append(lidx)
                            lag_selval_list[midx].append(laglist[lidx])
                    if (len(lag_selidx_list[midx]) > len(lag_selidx_list[max_laglen_idx])):
                        max_laglen_idx = midx
                    fout_list.append([BinaryImgs.OpenMIfileForWriting(aggr_root + vel_prefix + 'x' + strext),\
                                      BinaryImgs.OpenMIfileForWriting(aggr_root + vel_prefix + 'y' + strext)])
                    if (save_vsq):
                        fout_list[-1].append(BinaryImgs.OpenMIfileForWriting(aggr_root + vel_prefix + 'sq' + strext))
            
            time_start = time.time()
            for tidx in range(imgInfo[0]['img_num']):
                time_step = time.time()
                curt_mask = np.zeros((len(lag_selidx_list[max_laglen_idx]), imgInfo[2]['px_num']), dtype=bool)
                curt_dx = np.zeros((len(lag_selidx_list[max_laglen_idx]), imgInfo[2]['px_num']), dtype=float)
                curt_dy = np.zeros((len(lag_selidx_list[max_laglen_idx]), imgInfo[2]['px_num']), dtype=float)
                curt_lags = np.zeros((len(lag_selidx_list[max_laglen_idx]), imgInfo[2]['px_num']), dtype=float)
                for lidx in range(len(lag_selidx_list[max_laglen_idx])):
                    # mask==1: invalid vector
                    curt_mask[lidx] = BinaryImgs.getSingleImage_MIfile(res_files[2][lag_selidx_list[max_laglen_idx][lidx]], imgInfo[2], tidx, flatten_image=True)
                    curt_dx[lidx] = BinaryImgs.getSingleImage_MIfile(res_files[0][lag_selidx_list[max_laglen_idx][lidx]], imgInfo[0], tidx, flatten_image=True)
                    curt_dy[lidx] = BinaryImgs.getSingleImage_MIfile(res_files[1][lag_selidx_list[max_laglen_idx][lidx]], imgInfo[1], tidx, flatten_image=True)
                for midx in range(len(lag_selidx_list)):
                    if (fout_list[midx] is not None):
                        slopes = np.ones((imgInfo[2]['px_num'], 2), dtype=float) * np.nan
                        for pidx in range(imgInfo[2]['px_num']):
                            slopes[pidx,0] = SharedFunctions.LinearFit(lag_selval_list[midx], curt_dx[lag_selidx_list[midx],pidx], return_residuals=False, mask=curt_mask[:,pidx], catchex=False, nonan=True)
                            slopes[pidx,1] = SharedFunctions.LinearFit(lag_selval_list[midx], curt_dy[lag_selidx_list[midx],pidx], return_residuals=False, mask=curt_mask[:,pidx], catchex=False, nonan=True)
                        BinaryImgs.AppendToMIfile(fout_list[midx][0], slopes[:,0], 'f')
                        BinaryImgs.AppendToMIfile(fout_list[midx][1], slopes[:,1], 'f')
                        if (save_vsq):
                            BinaryImgs.AppendToMIfile(fout_list[midx][2], np.add(np.square(slopes[:,0]),np.square(slopes[:,1])), 'f')
                time_end = time.time()
                print('[{0}/{1}] Data for time {2} processed in {3:.1f} s. Elapsed time: {4:.1f} s'.format(tidx+1, imgInfo[0]['img_num'],\
                                                                          tidx, time_end-time_step, time_end-time_start))
            
            for midx in range(len(lag_selidx_list)):
                if (fout_list[midx] is not None):
                    for fidx in range(len(fout_list[midx])):
                        fout_list[midx][fidx].close()
            for fidx in range(len(res_files)):
                for lidx in range(len(res_files[fidx])):
                    res_files[fidx][lidx].close()
            
        else:
            
            raise ValueError('Postprocessing method ' + vel_method + ' not recognized')

        vel_filenames = [SharedFunctions.FindFileNames(aggr_root, Prefix=vel_prefix+'x', FilterString=vel_method, Ext='.raw'),\
                         SharedFunctions.FindFileNames(aggr_root, Prefix=vel_prefix+'y', FilterString=vel_method, Ext='.raw')]
        # TODO: go through velocity by lagtimes, starting from high, and figure out what's the best lagtime to use
    
    if ('-skipplot' in cmd_list or plot_root is None):
        
        print('Skipping plot step')
        
    else:
        
        SharedFunctions.CheckCreateFolder(plot_root)
        
        # Load refined velocities
        refinedv_filenames = [SharedFunctions.FindFileNames(aggr_root, Prefix=refinedv_prefix+'x', Ext='.raw'),\
                              SharedFunctions.FindFileNames(aggr_root, Prefix=refinedv_prefix+'y', Ext='.raw')]
        if (len(refinedv_filenames[0]) <= 0):
            raise ValueError('No ' + str(refinedv_prefix+'x') + '*.raw file in folder ' + str(aggr_root))
        if (len(refinedv_filenames[1]) <= 0):
            raise ValueError('No ' + str(refinedv_prefix+'y') + '*.raw file in folder ' + str(aggr_root))
        imgInfo = BinaryImgs.MIinfoFromName(refinedv_filenames[0][0], byteFormat='f')
        imgInfo['hdr_size'] = 0
        refinedv_data = BinaryImgs.ReadMIfileList([aggr_root+refinedv_filenames[0][0],aggr_root+refinedv_filenames[1][0]], MI_info=imgInfo, asArray=True)
        
        # Image sequence (just to get image dimensions and number of images)
        imSeq = ImageSequence(froot+SharedFunctions.ConfigGet(config, 'input', 'filter_frameA', '', str)+'*'+img_ext)
        imShape = imSeq[0].shape
        piv_coords = openpiv.process.get_coordinates(image_size=imShape, window_size=settings.windowsizes[settings.iterations-1],\
                                        overlap=settings.overlap[settings.iterations-1])
        
        # background
        normalize_background = SharedFunctions.ConfigGet(config, 'plot', 'normalize_background', None)
        imbkg_arr = None
        if (normalize_background is not None):
            if (os.path.isfile(normalize_background)):
                imbkg = Image.open(normalize_background)
                imbkg_arr = np.array(imbkg)
        
        # Binning
        bin_xy = SharedFunctions.ConfigGet(config, 'plot', 'bin_xy', 1, int)
        bin_z = SharedFunctions.ConfigGet(config, 'plot', 'bin_z', 1, int)
        piv_coords_binned = [SharedFunctions.downsample2d(piv_coords[0], bin_xy),\
                             SharedFunctions.downsample2d(piv_coords[1], bin_xy)]
        refinedv_data_binned = [SharedFunctions.downsample3d(refinedv_data[0], bin_xy, bin_z),\
                                SharedFunctions.downsample3d(refinedv_data[1], bin_xy, bin_z)]
        imSeq_binned = SharedFunctions.downsample3d(imSeq, 1, bin_z, norm=imbkg_arr, norm_type='2D')
        refinedv_data_binz = [SharedFunctions.downsample3d(refinedv_data[0], 1, bin_z),\
                              SharedFunctions.downsample3d(refinedv_data[1], 1, bin_z)]

        # Compute divergence and related options
        if (SharedFunctions.ConfigGet(config, 'plot', 'plot_divergence', False, bool)):
            diverg3D = np.add(np.gradient(refinedv_data_binz[0])[2], np.gradient(refinedv_data_binz[1])[1])
            divalphaexp = SharedFunctions.ConfigGet(config, 'plot', 'div_alpha_pwr', 0.0, float)
            divmaxalpha = np.power(np.nanmax(np.absolute(diverg3D[:-3])), divalphaexp)
            div_alpha_normbounds = np.multiply(divmaxalpha, SharedFunctions.ConfigGet(config, 'plot', 'div_alpha_normbounds', [0.0, 1.0], float))
            div_alphaclip = SharedFunctions.ConfigGet(config, 'plot', 'div_alphaclip', [0.0, 1.0], float)
            div_cmap = plt.cm.get_cmap(SharedFunctions.ConfigGet(config, 'plot', 'div_cmap', 'hot', str))
            div_cmap_bounds = [0, np.nanmax(diverg3D[:-3])]
        else:
            diverg3D = None

        if (SharedFunctions.ConfigGet(config, 'plot', 'plot_curl', False, bool)):
            curl3D = np.subtract(np.gradient(refinedv_data_binz[1])[2], np.gradient(refinedv_data_binz[0])[1])
            curl_alphaexp = SharedFunctions.ConfigGet(config, 'plot', 'curl_alpha_pwr', 0.0, float)
            curl_maxalpha = np.power(np.nanmax(np.absolute(curl3D[:-7])), curl_alphaexp)
            curl_alpha_normbounds = np.multiply(divmaxalpha, SharedFunctions.ConfigGet(config, 'plot', 'curl_alpha_normbounds', [0.0, 1.0], float))
            curl_alphaclip = SharedFunctions.ConfigGet(config, 'plot', 'curl_alphaclip', [0.0, 1.0], float)
            curl_cmap = plt.cm.get_cmap(SharedFunctions.ConfigGet(config, 'plot', 'curl_cmap', 'cool_r', str))
            curl_cmap_bounds = [np.nanmin(np.absolute(curl3D[:-7])), np.nanmax(np.absolute(curl3D[:-7]))]
        else:
            curl3D = None
        
        # Save all frames
        for tidx in range(min(len(imSeq), len(refinedv_data_binned[0]))):
            fig, ax = plt.subplots(figsize=(10,10))
            plot_vbounds = SharedFunctions.ConfigGet(config, 'plot', 'plot_vbounds', [np.min(imSeq_binned[tidx]), np.max(imSeq_binned[tidx])], float)
            ax.imshow(imSeq_binned[tidx], extent=[0, imShape[0], 0, imShape[1]], origin='upper', vmin=plot_vbounds[0], vmax=plot_vbounds[1], cmap='Greys_r')
            if (SharedFunctions.ConfigGet(config, 'plot', 'plot_quiver', False, bool)):
                ax.quiver(piv_coords_binned[0], piv_coords_binned[1], refinedv_data_binned[0][tidx], -refinedv_data_binned[1][tidx], color='blue', linewidth=2, scale=1)
            if (SharedFunctions.ConfigGet(config, 'plot', 'plot_streamlines', False, bool)):
                x_str, y_str = np.unique(piv_coords[0].flatten())[::bin_xy], imShape[1] - np.unique(piv_coords[1].flatten())[::bin_xy]
                u_str, v_str = refinedv_data_binned[0][tidx], -refinedv_data_binned[1][tidx]
                scalar_str = np.power(np.add(np.square(u_str), np.square(v_str)), 0.5)
                str_lw = 100*scalar_str #/ scalar_str.max()
                ax.streamplot(x_str, y_str, u_str, v_str,\
                              density=[1,0.6], linewidth=str_lw, color='b', arrowsize=0.2, minlength=0.04)#, cmap='hot'
            if (diverg3D is not None):
                cur_divdata = diverg3D[tidx]
                divalphas = Normalize(vmin=div_alpha_normbounds[0], vmax=div_alpha_normbounds[1], clip=True)(np.power(np.absolute(np.nan_to_num(cur_divdata)), divalphaexp))
                divalphas = np.clip(divalphas, div_alphaclip[0], div_alphaclip[1])
                divcolors = Normalize(vmin=div_cmap_bounds[0], vmax=div_cmap_bounds[1])(cur_divdata) # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                divcolors = div_cmap(divcolors)
                divcolors[..., -1] = divalphas
                ax.imshow(divcolors, extent=[0, imShape[0], 0, imShape[1]], origin='upper')
            if (curl3D is not None):
                cur_curldata = curl3D[tidx]
                curlalphas = Normalize(vmin=curl_alpha_normbounds[0], vmax=curl_alpha_normbounds[1], clip=True)(np.power(np.absolute(np.nan_to_num(cur_curldata)), curl_alphaexp))
                curlalphas = np.clip(curlalphas, curl_alphaclip[0], curl_alphaclip[1])
                curlcolors = Normalize(vmin=curl_cmap_bounds[0], vmax=curl_cmap_bounds[1])(cur_curldata) # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                curlcolors = curl_cmap(curlcolors)
                curlcolors[..., -1] = curlalphas
                ax.imshow(curlcolors, extent=[0, imShape[0], 0, imShape[1]], origin='upper')                
            ax.set_position([0, 0, 1, 1])
            plt.axis('off')
            fig.savefig(plot_root + str(tidx).zfill(4) + '.png')
            plt.close('all')