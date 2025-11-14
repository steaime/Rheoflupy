import os
import time
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import Rheoflu.IOfunctions as iof
import Rheoflu.ChannelShapeAnalysis as csa
import Rheoflu.DropletAnalysis as da

def rheoflu_analysis(param_fpath, **kwargs):
    
    if os.path.isfile(param_fpath):
        with open(param_fpath, 'r') as f:
            params = json.load(f)
            print('Analysis parameters loaded from configuration file: ' + param_fpath)
        for k, val in kwargs.items():
            if k in params:
                print('param[{0}] updated from {1} to {2} using function kwargs'.format(k, params[k], kwargs[k]))
            else:
                print('param[{0}]={1} added using function kwargs'.format(k, kwargs[k]))
            params[k] = kwargs[k]
            
        #################
        froot = params['froot']
        img_name = params['img_name']
        bkg_avgrange = params['bkg_avgrange']
        #################
        print('\n### 1: LOAD IMAGE SEQUENCE')
        print('Working in root folder ' + froot)
        print('Loading image ' + img_name + '.tif')
        fpath = os.path.join(froot, img_name + '.tif')
        print('Image stack has shape: {0}'.format(iof.get_stack_shape(fpath)))
        t0 = time.time()
        bkg = iof.compute_background(fpath, avg_range=bkg_avgrange)
        print('Computing z average took {0:.2f} seconds'.format(time.time()-t0))
        
        ##########
        px_size = params['px_size']
        fps = params['fps']
        eta = params['eta']
        design_omega = params['design_omega']
        topedge_fpath = params['topedge_fpath']
        bottomedge_fpath = params['bottomedge_fpath']
        ##########
        print('\n### 2: ANALYZE CHANNEL SHAPE')
        ch_params, shape, edges_um, chaxis_px, minpos_px = csa.AnalyzeChannelShape(bkg, bottomedge_fpath, topedge_fpath,
                                                px_size=px_size, design_omega=design_omega, eta=eta, save_fig=os.path.join(froot, img_name + '_chshape.png'))
        print('constriction positions [px]: ' + str(minpos_px))
        print('constriction positions [µm]: ' + str([pos*px_size for pos in minpos_px]))
        print('channel axis position [px]: ' + str(chaxis_px))
        
        ##########
        filter_d = params['filter_d']
        crop_margin_x = params['crop_margin_x']
        crop_size_y = params['crop_size_y']
        track_minmass = params['track_minmass']
        track_maxsize = params['track_maxsize']
        track_nprocs = params['track_nprocs']
        search_range = params['search_range']
        filter_range = params['filter_range']
        bkgcorr_off = params['bkgcorr_off']
        track_zRange = params['track_zRange']
        roi_test_frame = params['roi_test_frame']
        filter_ss_maxtomean = params['filter_ss_maxtomean']
        ##########
        print('\n### 3: TRACK DROPLETS AND COMPUTE STRESSES')
        out_root = os.path.join(froot, 'out_d' + str(filter_d))
        iof.CheckCreateFolder(out_root)
        if track_zRange is None:
            track_zRange = [0, iof.get_stack_shape(fpath)[0]]
        crop_ROI = da.get_track_roi(minpos_px, chaxis_px, crop_margin_x, crop_size_y, filter_range=filter_range, fpath=fpath, test_frame=roi_test_frame, 
                    bkg=bkg, bkgcorr_off=bkgcorr_off, filter_d=filter_d, minmass=track_minmass, save_fig=os.path.join(out_root, img_name+'_FIG1_trackROI.png'))
        
        df_savepath = os.path.join(out_root, img_name+'_track.csv')
        force_track = False
        if os.path.isfile(df_savepath) and not force_track:
            print('csv filename with track result already present: skip track calculation')
            track_df = pd.read_csv(df_savepath)
            PID_list = track_df['particle'].unique()
        else:
            track_filter = [[minpos_px[0]-filter_range[0], minpos_px[1]+filter_range[0]],
                            [chaxis_px-filter_range[1], chaxis_px+filter_range[1]]]
            fstack = iof.get_stack(fpath, track_zRange, cropROI=crop_ROI, bkg=bkg, bkgcorr_offset=bkgcorr_off)
            track_df, PID_list = da.track_droplets(fstack, stack_offset=[track_zRange[0], crop_ROI[1], crop_ROI[0]], diameter=filter_d, minmass=track_minmass, 
                              maxsize=track_maxsize, filter_range=track_filter, search_range=search_range, link_memory=0, 
                                track_out_fpath=os.path.join(out_root, img_name+'.h5'), track_procs=track_nprocs, df_savepath=df_savepath)
        da.plot_trajectories(track_df, bkg_img=bkg, PID_list=PID_list, chaxis_px=chaxis_px, constr_pos=minpos_px, filter_range=filter_range,
                            save_fig=os.path.join(out_root, img_name+'_FIG2_trajectories.png'))
        PID_sel = da.track_postproc(track_df, px_size=px_size, fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot=True, verbose=0, x_off=minpos_px[0]*px_size, 
                                   save_fig=os.path.join(out_root, img_name+'_FIG3_speedstress.png'))
        
        ##########
        edge_roi_size = params['edge_roi_size']
        edge_blur = params['edge_blur']
        edge_dbscan_eps = params['edge_dbscan_eps']
        edge_thr = params['edge_thr']
        edge_iterN = params['edge_iterN']
        filter_r_thr = params['filter_r_thr']
        save_all_edgefigs = params['save_all_edgefigs']
        ##########
        print('\n### 4: MEASURE DROPLET DEFORMATION')
        defdf_savepath = os.path.join(out_root, img_name+'_def.csv')
        force_def = False
        if os.path.isfile(defdf_savepath) and not force_def:
            print('csv filename with deformation result already present: skip deformation calculation')
            track_df = pd.read_csv(defdf_savepath)
            PID_sel = track_df['particle'].unique()
        else:
            if save_all_edgefigs:
                edges_plot_outdir = os.path.join(froot, 'out_tmp')
            else:
                edges_plot_outdir = None
            def_res, PID_sel = da.analyze_deformations(fpath, track_df, crop_roi_size=edge_roi_size, img_bkg=bkg, img_bkgcorr_offset=bkgcorr_off, 
                                                       img_blur_sigma=edge_blur, edge_threshold=edge_thr, smoothing_iterN=edge_iterN, 
                                                       dbscan_eps=edge_dbscan_eps, dbscan_minN=1, filter_r_thr=filter_r_thr, 
                                                       plot_outdir=edges_plot_outdir, px_size=px_size, fps=fps, PID_list=PID_sel)
            track_df.to_csv(defdf_savepath)
        ##########
        filter_rbar_relstd = params['filter_rbar_relstd']
        filter_rtheta_mserr = params['filter_rtheta_mserr']
        ##########
        PID_sel = da.filter_droplets(track_df, thr_relstd=filter_rbar_relstd, thr_mserr=filter_rtheta_mserr, PID_list=PID_sel)
        
        ########
        ss_fit_margin = params['ss_fit_margin']
        ss_param_bound = params['ss_param_bound']
        ss_globalfit_iter = params['ss_globalfit_iter']
        ########
        print('\n### 5: DROPLET RHEOLOGY')
        pID_subset_stress = da.track_postproc(track_df, px_size=px_size, fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot=False, verbose=0)
        PID_sel = [pID for pID in PID_sel if pID in pID_subset_stress]
        moduli_csv_fpath = os.path.join(out_root, img_name+'_moduli.csv')
        moduli_img_fpath = os.path.join(out_root, img_name+'_FIG5_moduli.png')
        moduli_df = da.calc_moduli(track_df, PID_list=PID_sel, fps=fps, fit_margin=ss_fit_margin, param_bound=ss_param_bound, 
                                   pre_fit_iter=ss_globalfit_iter, plot=True, save_csv=moduli_csv_fpath, save_fig=moduli_img_fpath)
        
        LB_img_fpath = os.path.join(out_root, img_name+'_FIG4_LB.png')
        PID_sel_sorted = da.plot_lissajous(track_df, pID_subset=PID_sel, mod_df=moduli_df, recalc_stress=True, px_size=px_size, 
                                    fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot_alpha=0.3, save_fig=LB_img_fpath)
        
        params_fpath = os.path.join(out_root, img_name+'_analysisParams.log')
        print('\nAnalysis completed in {0:.2f} seconds. Parameters saved to file {1}'.format(time.time()-t0, params_fpath))
        with open(params_fpath, 'w') as fout:
            json.dump(params, fout, indent=2)