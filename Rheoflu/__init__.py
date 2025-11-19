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
    
    params = iof.load_params(param_fpath, kwargs)
    
    #################
    froot               = params['froot']
    img_name            = params['img_name']
    bkg_avgrange        = params['bkg_avgrange']
    px_size             = params['px_size']
    fps                 = params['fps']
    eta                 = params['eta']
    design_omega        = params['design_omega']
    topedge_fpath       = params['topedge_fpath']
    bottomedge_fpath    = params['bottomedge_fpath']
    filter_d            = params['filter_d']
    crop_margin_x       = params['crop_margin_x']
    crop_size_y         = params['crop_size_y']
    track_minmass       = params['track_minmass']
    track_maxsize       = params['track_maxsize']
    track_nprocs        = params['track_nprocs']
    search_range        = params['search_range']
    filter_range        = params['filter_range']
    bkgcorr_off         = params['bkgcorr_off']
    track_zRange        = params['track_zRange']
    roi_test_frame      = params['roi_test_frame']
    filter_ss_maxtomean = params['filter_ss_maxtomean']
    allowed_badpoints   = params['allowed_badpoints']
    edge_roi_size       = params['edge_roi_size']
    edge_blur           = params['edge_blur']
    edge_dbscan_eps     = params['edge_dbscan_eps']
    edge_thr            = params['edge_thr']
    edge_iterN          = params['edge_iterN']
    filter_r_thr        = params['filter_r_thr']
    save_all_edgefigs   = params['save_all_edgefigs']
    filter_rbar_relstd  = params['filter_rbar_relstd']
    filter_rtheta_mserr = params['filter_rtheta_mserr']
    ss_fit_margin       = params['ss_fit_margin']
    ss_param_bound      = params['ss_param_bound']
    ss_globalfit_iter   = params['ss_globalfit_iter']
    #################
    
    fpath = os.path.join(froot, img_name + '.tif')
    out_root = os.path.join(froot, img_name + '_out_d' + str(filter_d))
    iof.CheckCreateFolder(out_root)
    flog = iof.setup_logger(os.path.join(out_root, img_name + '_analysis_log.txt'))
    t0 = time.time()
    
    
    
    iof.printlog('\n### 1: LOAD IMAGE SEQUENCE\n  Working in root folder {0}\n  Loading image {1}.tif\n  Image stack has shape: {2}'.format(froot, img_name, iof.get_stack_shape(fpath)), flog)
    bkg = iof.compute_background(fpath, avg_range=bkg_avgrange)
    iof.printlog('Computing z average took {0:.2f} seconds'.format(time.time()-t0), flog)

    
    
    iof.printlog('\n### 2: ANALYZE CHANNEL SHAPE', flog)
    if topedge_fpath=='auto':
        topedge_fpath = os.path.join(froot, img_name + '_edge1_px.txt')
        iof.printlog('top edge file path automatically generated: ' + topedge_fpath, flog)
    if bottomedge_fpath=='auto':
        bottomedge_fpath = os.path.join(froot, img_name + '_edge2_px.txt')
        iof.printlog('bottom edge file path automatically generated: ' + topedge_fpath, flog)
    ch_params, shape, edges_um, chaxis_px, minpos_px = csa.AnalyzeChannelShape(bkg, bottomedge_fpath, topedge_fpath,
                                            px_size=px_size, design_omega=design_omega, eta=eta, save_fig=os.path.join(froot, img_name + '_chshape.png'))
    iof.printlog('Shape analysis result:\n  - constriction positions [px]: {0}\n  - constriction positions [µm]: {1}\n  - channel axis position [px]: {2}'.format(minpos_px, [pos*px_size for pos in minpos_px], chaxis_px), flog)

    
    
    iof.printlog('\n### 3: TRACK DROPLETS AND COMPUTE STRESSES', flog)
    if track_zRange is None:
        track_zRange = [0, iof.get_stack_shape(fpath)[0]]
    if track_zRange[1] > iof.get_stack_shape(fpath)[0]:
        track_zRange[1] = iof.get_stack_shape(fpath)[0]
    crop_ROI = da.get_track_roi(minpos_px, chaxis_px, crop_margin_x, crop_size_y, filter_range=filter_range, fpath=fpath, test_frame=roi_test_frame, 
                bkg=bkg, bkgcorr_off=bkgcorr_off, filter_d=filter_d, minmass=track_minmass, save_fig=os.path.join(out_root, img_name+'_FIG1_trackROI.png'))

    df_savepath = os.path.join(out_root, img_name+'_track.csv')
    force_track = False
    if os.path.isfile(df_savepath) and not force_track:
        iof.printlog('csv filename with track result already present: skip track calculation', flog)
        track_df = pd.read_csv(df_savepath)
        PID_list = track_df['particle'].unique()
    else:
        track_filter = [[minpos_px[0]-filter_range[0], minpos_px[1]+filter_range[0]],
                        [chaxis_px-filter_range[1], chaxis_px+filter_range[1]]]
        fstack = iof.get_stack(fpath, track_zRange, cropROI=crop_ROI, bkg=bkg, bkgcorr_offset=bkgcorr_off)
        track_df, PID_list = da.track_droplets(fstack, stack_offset=[track_zRange[0], crop_ROI[1], crop_ROI[0]], diameter=filter_d, minmass=track_minmass, 
                          maxsize=track_maxsize, filter_range=track_filter, search_range=search_range, link_memory=0, 
                            track_out_fpath=os.path.join(out_root, img_name+'.h5'), track_procs=track_nprocs, df_savepath=df_savepath, flog=flog)
    da.plot_trajectories(track_df, bkg_img=bkg, PID_list=PID_list, chaxis_px=chaxis_px, constr_pos=minpos_px, filter_range=filter_range,
                        save_fig=os.path.join(out_root, img_name+'_FIG2_trajectories.png'), flog=flog)
    PID_sel = da.track_postproc(track_df, px_size=px_size, fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot=True, 
                                verbose=0, x_off=minpos_px[0]*px_size, save_fig=os.path.join(out_root, img_name+'_FIG3_speedstress.png'), flog=flog)

    
    
    iof.printlog('\n### 4: MEASURE DROPLET DEFORMATION', flog)
    defdf_savepath = os.path.join(out_root, img_name + '_def.csv')
    force_def = False
    if os.path.isfile(defdf_savepath) and not force_def:
        iof.printlog('csv filename with deformation result already present: skip deformation calculation', flog)
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
                                                   plot_outdir=edges_plot_outdir, px_size=px_size, fps=fps, PID_list=PID_sel, flog=flog)
        track_df.to_csv(defdf_savepath)
    filter_img_fpath = os.path.join(out_root, img_name+'_FIG4_filter_droplets.png')
    PID_sel = da.filter_droplets(track_df, thr_relstd=filter_rbar_relstd, thr_mserr=filter_rtheta_mserr, 
                                 allowed_badpoints=allowed_badpoints, PID_list=PID_sel, save_fig=filter_img_fpath, flog=flog)

    
    
    iof.printlog('\n### 5: DROPLET RHEOLOGY', flog)
    pID_subset_stress = da.track_postproc(track_df, px_size=px_size, fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot=False, verbose=0, flog=flog)
    PID_sel = [pID for pID in PID_sel if pID in pID_subset_stress]
    moduli_csv_fpath = os.path.join(out_root, img_name+'_moduli.csv')
    moduli_img_fpath = os.path.join(out_root, img_name+'_FIG6_moduli.png')
    moduli_df = da.calc_moduli(track_df, PID_list=PID_sel, fps=fps, fit_margin=ss_fit_margin, param_bound=ss_param_bound, 
                               pre_fit_iter=ss_globalfit_iter, plot=True, save_csv=moduli_csv_fpath, save_fig=moduli_img_fpath, flog=flog)

    LB_img_fpath = os.path.join(out_root, img_name+'_FIG5_LB.png')
    PID_sel_sorted = da.plot_lissajous(track_df, pID_subset=PID_sel, mod_df=moduli_df, recalc_stress=True, px_size=px_size, 
                                fps=fps, eta=eta, ss_maxtomean=filter_ss_maxtomean, plot_alpha=0.3, save_fig=LB_img_fpath)

    
    
    params_fpath = os.path.join(out_root, img_name+'_analysisParams.log')
    iof.printlog('Analysis completed in {0:.2f} seconds. Parameters saved to file {1}'.format(time.time()-t0, params_fpath), flog)
    with open(params_fpath, 'w') as fout:
        json.dump(params, fout, indent=2)
    iof.close_logger(flog)