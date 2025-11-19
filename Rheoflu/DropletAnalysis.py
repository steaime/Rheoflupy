import os
import time
import shutil
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.signal import argrelmax
from scipy.optimize import curve_fit, least_squares
from sklearn.cluster import DBSCAN
from subpixel_edges import subpixel_edges
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import Rheoflu.IOfunctions as iof

def get_track_roi(minpos_px, chaxis_px, crop_margin_x, crop_size_y, filter_range=None, fpath=None, test_frame=1, 
                  bkg=None, bkgcorr_off=0, filter_d=None, minmass=None, save_fig=None):
    crop_ROI = [minpos_px[0] - crop_margin_x, chaxis_px - crop_size_y//2,
            minpos_px[1] + crop_margin_x, chaxis_px + crop_size_y//2]
    if fpath is not None:
        full_frame = iof.get_single_frame(fpath, test_frame)
        frame = iof.get_single_frame(fpath, test_frame, cropROI=crop_ROI, bkg=bkg, bkgcorr_offset=bkgcorr_off)
        f_locate = tp.locate(frame, filter_d, minmass=minmass)
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(full_frame, cmap='Greys_r', extent=(0, full_frame.shape[1], full_frame.shape[0], 0))
        ax[0].add_patch(patches.Rectangle((crop_ROI[0], crop_ROI[1]), crop_ROI[2]-crop_ROI[0], crop_ROI[3]-crop_ROI[1], edgecolor='r', facecolor='none'))
        ax[0].add_patch(patches.Rectangle((minpos_px[0], chaxis_px-1), minpos_px[1]-minpos_px[0], 2, edgecolor='k', facecolor='none'))
        tp.annotate(f_locate, frame, ax=ax[1], plot_style={'markersize': filter_d})
        ax[1].add_patch(patches.Rectangle((crop_margin_x, crop_size_y//2), minpos_px[1]-minpos_px[0], 2, edgecolor='k', facecolor='none'))
        if filter_range is not None:
            ax[0].add_patch(patches.Rectangle((minpos_px[0]-filter_range[0], chaxis_px-filter_range[1]), 
                                              minpos_px[1]-minpos_px[0]+2*filter_range[0], 2*filter_range[1], edgecolor='y', facecolor='none'))
            ax[1].add_patch(patches.Rectangle((crop_margin_x-filter_range[0], crop_size_y//2-filter_range[1]), 
                                              minpos_px[1]-minpos_px[0]+2*filter_range[0], 2*filter_range[1], edgecolor='y', facecolor='none'))
        if save_fig is not None:
            fig.savefig(save_fig)
    return crop_ROI

def track_droplets(img_stack, stack_offset, diameter, minmass, search_range, track_out_fpath, filter_range=None, maxsize=None, 
                   link_memory=0, track_procs=4, df_savepath=None, clean_after=True, flog=None):
    with tp.PandasHDFStore(track_out_fpath) as s:
        tp.batch(img_stack, diameter, minmass=minmass, processes=4, output=s)
        # As before, we require a minimum "life" of 5 frames and a memory of 3 frames
        for linked in tp.link_df_iter(s, search_range=search_range, memory=link_memory, link_strategy='auto'):
            s.put(linked)
        t = pd.concat(iter(s))
        
    if clean_after:
        os.remove(track_out_fpath)
    
    track_df = t.copy()
    track_df['y'] = t['y'] + stack_offset[1]
    track_df['x'] = t['x'] + stack_offset[2]
    track_df['frame'] = t['frame'] + stack_offset[0]
    
    if filter_range is not None:
        groups = track_df.groupby('particle')
        particles_in_range = []
        for name, group in groups:
            if group['x'].min() <= filter_range[0][0] and \
                group['x'].max() >= filter_range[0][1] and \
                group['y'].min() >= filter_range[1][0] and \
                group['y'].max() <= filter_range[1][1]:
                particles_in_range.append(name)
        track_df = track_df[track_df['particle'].isin(particles_in_range)]
        iof.printlog('Filtering trajectories reduced dataset from {0} to {1} particles'.format(len(t['particle'].unique()), len(track_df['particle'].unique())), flog)
        
    if df_savepath is not None:
        track_df.to_csv(df_savepath)
    PID_list = track_df['particle'].unique()
    
    return track_df, PID_list

def plot_trajectories(track_df, bkg_img, PID_list=None, chaxis_px=None, constr_pos=None, filter_range=None, save_fig=None, flog=None):
    if PID_list is None:
        PID_list = track_df['particle'].unique()
    iof.printlog('{0} particles have been tracked across the whole constriction'.format(len(PID_list)), flog)
    fig, ax = plt.subplots(nrows=2)
    tp.plot_traj(track_df, superimpose=bkg_img, ax=ax[0], plot_style={'linewidth': 2})
    for pID in PID_list:
        ax[1].plot(track_df[(track_df['particle']==pID)]['x'], track_df[(track_df['particle']==pID)]['y'])
    if chaxis_px is not None:
        ax[1].axhline(chaxis_px, color='k', ls='--')
    if constr_pos is not None:
        for cur_pos in constr_pos:
            ax[1].axvline(cur_pos, color='k', ls=':')
        if chaxis_px is not None and filter_range is not None:
            ax[0].add_patch(patches.Rectangle((constr_pos[0]-filter_range[0], chaxis_px-filter_range[1]), 
                                      constr_pos[1]-constr_pos[0]+2*filter_range[0], 2*filter_range[1], edgecolor='y', facecolor='none'))
    if save_fig is not None:
        fig.savefig(save_fig)

            
def calc_droplet_stress(x_px, px_size, fps, eta, verbose=0, params=None, pID=0, plot=False, flog=None):
    x_pos = np.array(px_size * x_px)
    v = np.gradient(x_pos, 1./fps) #um/s
    dvdx = np.gradient(v, x_pos)
    stress = eta*dvdx

    if verbose>1:
        maxidx = argrelmax(v)[0]
        if len(maxidx)>1:
            period = (maxidx[1]-maxidx[0])/fps
        else:
            iof.printlog('WARNING: unable to estimate period from flow speed', flog)
            period = np.nan
        omega = 2*np.pi/period
        vmax = np.max(v)
        strmsg =    '  +----------------------------------------------+'
        strmsg += '\n  |       TRACE ANALYSIS FOR DROPLET {0:05d}:      |'.format(pID)
        strmsg += '\n  +----------------------------------------------+'
        strmsg += '\n  | Maximum droplet speed:  vmax ={0:8.2f} mm/s  |'.format(vmax*1e-3)
        strmsg += '\n  | Frequency:             omega ={0:8.1f} rad/s |'.format(omega)
        if params is not None:
            q_est = params['L0']*vmax*1e-6
            strmsg += '\n  | Planar flow rate (est.):   q ={0:8.2f} mm2/s |'.format(q_est*1e6)
            strmsg += '\n  | Reduced frequency (exp)  w/q ={0:8.2f} 1/mm2 |'.format(omega/(q_est*1e6))
            strmsg += '\n  | Reduced frequency (params)   ={0:8.2f} 1/mm2 |'.format(params['omega']/(params['q']*1e6))
            strmsg += '\n  | Real stress amplitude: sigma ={0:8.1f} Pa    |'.format(params['stress_amp']/params['omega']*omega)
            strmsg += '\n  | Stress amplitude (params): s ={0:8.1f} Pa    |'.format(params['stress_amp'])
        strmsg += '\n  +----------------------------------------------+'
        iof.printlog(strmsg, flog)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x_pos, v*1e-3, 'k-')
        ax2 = ax.twinx()
        ax2.plot(x_pos, stress, 'r:')
        ax.set_ylabel(r'$v$ [mm/s]')
        ax.set_xlabel(r'$x$ [µm]')
        ax2.set_ylabel(r'$\sigma$ [Pa]')
        
    return x_pos*1e-6, v*1e-6, stress

def track_postproc(track_df, px_size, fps, eta, ss_maxtomean=None, save_fname=None, plot=True, verbose=0, x_off=0, params=None, update_df=True, save_fig=None, flog=None):
    if plot:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
    if save_fname is not None:
        res_data = []
        str_hdr = ''        
    PID_list = track_df['particle'].unique()
    PID_sel = []
    max_dx, max_dy = [], []
    count_excluded = 0
    for cur_PID in PID_list:
        cur_xarr = track_df[(track_df['particle'] == cur_PID)]['x']
        x_pos, v, stress = calc_droplet_stress(cur_xarr, px_size=px_size, fps=fps, eta=eta, verbose=verbose, params=params)
        if update_df:
            track_df.loc[(track_df['particle'] == cur_PID), 'vx'] = v
            track_df.loc[(track_df['particle'] == cur_PID), 'stress'] = stress
        if ss_maxtomean is not None:
            drop_ok = np.max(np.square(stress))/np.mean(np.square(stress)) < ss_maxtomean
        else:
            drop_ok = True
        if drop_ok:
            max_dx.append(np.max(np.diff(cur_xarr)))
            max_dy.append(np.max(np.diff(track_df[(track_df['particle'] == cur_PID)]['y'])))
            PID_sel.append(cur_PID)
            x_pos *= 1e6
            v *= 1e6
            if plot:
                ax.plot(x_pos-x_off, v*1e-3, ls=':')
                ax2.plot(x_pos-x_off, stress)
            if save_fname is not None:
                res_data.append(x_pos-x_off)
                res_data.append(v*1e-3)
                res_data.append(stress)
                if str_hdr != '':
                    str_hdr += '\t'
                str_hdr += 'x_{0}[um]\tv_{0}[mm/s]\ts_{0}[Pa]'.format(cur_PID)
        else:
            count_excluded += 1
    if plot:
        ax.set_ylabel(r'$v$ [mm/s]')
        ax.set_xlabel(r'$x-x_0$ [µm]')
        ax2.set_ylabel(r'$\sigma$ [Pa]')
        if params is not None:
            ax.set_xlim([0, params['wavelength_um']*1e6])
        if save_fig is not None:
            fig.savefig(save_fig)
    if save_fname is not None:
        max_len = np.max([len(x) for x in res_data])
        for i in range(len(res_data)):
            for j in range(max_len-len(res_data[i])):
                res_data[i] = np.append(res_data[i], np.nan)
        np.savetxt(os.path.join(froot, img_name+'_stress.dat'), np.array(res_data).T, delimiter='\t', header=str_hdr)
    if count_excluded>0:
        iof.printlog('{0}/{1} droplets excluded from the analysis'.format(count_excluded, len(PID_list)), flog)
    if len(max_dx)>0 and len(max_dy)>0:
        iof.printlog('Max displacement between adjacent frames: [{0:.1f}, {1:.1f}] pixels'.format(np.max(max_dx), np.max(max_dy)), flog)
    else:
        iof.printlog('WARNING: no selected droplets!', flog)
    return PID_sel

def drop_cropROI(track_df, pID, frame, roi_size, rel_frame=False, fpath=None, bkg=None, bkgcorr_offset=0, blur_sigma=0):
    if rel_frame:
        drop_frame = int(track_df[(track_df['particle']==pID)]['frame'].iloc[[frame]])
    else:
        drop_frame = frame
    roi = track_df[(track_df['particle']==pID) & (track_df['frame'] == drop_frame)]
    xloc, yloc = int(roi["x"].iloc[0]), int(roi["y"].iloc[0])
    drop_ROI = [xloc-roi_size, yloc-roi_size, xloc+roi_size, yloc+roi_size]
    if fpath is not None:
        imgcrop = iof.get_single_frame(fpath, drop_frame, cropROI=drop_ROI, bkg=bkg, bkgcorr_offset=bkgcorr_offset, blur_sigma=blur_sigma, dtype=float)
    else:
        imgcrop = None
    return drop_ROI, imgcrop
        
def find_edges(image, edge_threshold, smoothing_iterN, dbscan_eps=1, dbscan_minN=1, plot=False):
    drop_edges = subpixel_edges(image, edge_threshold, smoothing_iterN, 2)
    edge_points = np.array([drop_edges.x,drop_edges.y]).T
    out_edge, cluster_labels = extract_outer_edge(edge_points, dbscan_eps, dbscan_minN)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='Greys_r', extent=[0, image.shape[1], 0, image.shape[0]], origin='lower', aspect='auto')
        ax.imshow(np.ones_like(image), cmap='Greys_r', vmin=0, vmax=1, alpha=0.5, extent=[0, image.shape[1], 0, image.shape[0]], origin='lower', aspect='auto')
        ax.scatter(edge_points[:, 0], edge_points[:, 1], c=cluster_labels, cmap='Paired', alpha=0.5)
        ax.scatter(out_edge[:, 0], out_edge[:, 1], c='red', label='Selected edge points')
        ax.legend()
    return out_edge

def extract_outer_edge(edge_points, eps, min_samples, flog=None):
    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    y_pred = db.fit_predict(edge_points)
    unique_labels = set(y_pred)
    unique_labels.discard(-1)

    centroids, average_radii, cluster_data = [], [], []

    # Calculate centroids, average radii, and sizes for each cluster
    for label in unique_labels:
        points = edge_points[y_pred == label]
        cluster_size = len(points)
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)

        radii = np.linalg.norm(points - centroid, axis=1)
        average_radius = np.mean(radii)
        average_radii.append(average_radius)

        if cluster_size > 0:
            xc=np.mean(np.array(points[:,0]))
            yc=np.mean(np.array(points[:,1]))
        cluster_data.append([label, cluster_size, np.array([xc, yc])])

    # Sort clusters by size and take the top two largest clusters
    top_two_clusters = sorted(cluster_data, key=lambda x: x[1], reverse=True)[:2]

    tot_n = np.sum([x[1] for x in top_two_clusters])
    true_center = [x[2]*(x[1]/tot_n) for x in top_two_clusters]
    true_center = np.sum(true_center, axis=0)

    #add radius to the cluster data
    for i in range(len(top_two_clusters)):
        num_points = len(edge_points[y_pred == top_two_clusters[i][0]])
        top_two_clusters[i].append((np.linalg.norm(np.array(edge_points[y_pred == top_two_clusters[i][0]]) - true_center))/num_points)
    top_two_clusters.sort(key=lambda x: x[3], reverse=True)

    if len(top_two_clusters) < 1: 
        iof.printlog("Less than one clusters found.", flog)
        return None, None
    else:
        # Among the top two clusters, select the one with the smallest average radius
        outermost_label = top_two_clusters[0][0]
        # Extract the points that belong to the innermost circle
        outermost_points = edge_points[y_pred == outermost_label]
        return outermost_points, y_pred

def r_theta_circle(theta, r0, x0, y0):
    return r0 * (1 - x0*np.cos(theta) - y0*np.sin(theta))

def r_theta_ellipse(theta, r0, x0, y0, gamma):
    return r0 * (1 - x0*np.cos(theta) - y0*np.sin(theta) + gamma*np.cos(2*theta))

def r_theta_higherorder(theta, r0, x0, y0, g2, g3, g4, g5):
    return r0 * (1 - x0*np.cos(theta) - y0*np.sin(theta) + g2*np.cos(2*theta) + g3*np.cos(3*theta) + g4*np.cos(4*theta) + g5*np.cos(5*theta))

def calc_r2(y, yfit):
    return 1 - np.sum(np.square(y-yfit))/np.sum(np.square(y-np.mean(y)))

def calc_Pearson(covar):
    return np.sqrt(covar[1][0]*covar[0][1]/(covar[0][0]*covar[1][1]))

def fit_edge(drop_edges, guess_bound=0.1, filter_r_thr=None, frame_n=None, print_res=False, plot=True, plot_savedir=None, flog=None): # the edges are before rotation

    x, y = drop_edges[:,0], drop_edges[:,1]
    cx, cy = np.mean(x), np.mean(y)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    if filter_r_thr is None:
        #filter out r and theta value where r much greater than r_bar, only fit inner circle
        filt_idx = np.where(r <= np.mean(r) + filter_r_thr)
        x = x[filt_idx]
        y = y[filt_idx]
        # renew centers
        cx, cy = np.mean(x), np.mean(y)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    theta = np.arctan2(y - cy, x - cx)
    dx, dy = np.max(x)-np.min(x), np.max(y)-np.min(y)
    guess_rbar = np.mean(r)
    guess_g2 = (dx - dy)/(dx + dy)
    
    popt_crc, pcov_crc = curve_fit(r_theta_circle, theta, r, p0=[guess_rbar,0,0])
    guess_rbar, guess_xc, guess_yc = popt_crc[0], popt_crc[1], popt_crc[2]
    partial_ellipse_fitfunc = partial(r_theta_ellipse, r0=guess_rbar, x0=guess_xc, y0=guess_yc)
    popt_ell, pcov_ell = curve_fit(lambda theta, gamma: r_theta_ellipse(theta, *popt_crc, gamma), theta, r, p0=[guess_g2])
    guessp = [guess_rbar, guess_xc, guess_yc, popt_ell[0], 0, 0, 0]
    if guess_bound is not None:
        pbounds = ([guess_rbar*(1-guess_bound), guess_xc-1, guess_yc-1, 
                    popt_ell[0]-guess_bound, -guess_bound, -guess_bound, -guess_bound],
                   [guess_rbar*(1+guess_bound), guess_xc+1, guess_yc+1, 
                    popt_ell[0]+guess_bound, guess_bound, guess_bound, guess_bound])
        method = 'trf'
    else:
        pbounds = (-np.inf, np.inf)
        method = 'lm'
        
    try:
        popt, pcov = curve_fit(r_theta_higherorder, theta, r, method=method, p0=guessp, bounds=pbounds)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        iof.printlog('WARNING: fit with nonlinear droplet deformation did not converge. Second order parameter only', flog)
        popt, pcov = guessp, None
        perr = np.zeros_like(guessp)
    r_fit = r_theta_higherorder(theta, *popt)
    r_mserr = np.mean(np.square(r - r_fit))
    non_ellip = np.sum(np.abs(popt[4:]))/np.abs(popt[3])
    
    res = {'cx'    : cx,
           'cy'    : cy,
           'dx'    : dx,
           'dy'    : dy,
           'rbar'  : popt[0],
           'rb_er' : perr[0],
           'x0'    : popt[1],
           'x0_er' : perr[1],
           'y0'    : popt[2],
           'y0_er' : perr[2],
           'g2'    : popt[3],
           'g2_er' : perr[3],
           'g3'    : popt[4],
           'g3_er' : perr[4],
           'g4'    : popt[5],
           'g4_er' : perr[5],
           'g5'    : popt[6],
           'g5_er' : perr[6],
           'R2'    : calc_r2(r, r_fit),
           'mserr' : r_mserr,
           'rmean' : np.mean(r),
           'rawg2' : guess_g2,
           'r0_el' : guess_rbar,
           'x0_el' : guess_xc,
           'y0_el' : guess_yc,
           'g2_el' : popt_ell[0],
           'nonel' : non_ellip,
          }
    
    if plot:
        # Plot the reconstructed droplet shape
        theta_fit = np.linspace(-np.pi, np.pi, 1000)
        r_fit = r_theta_higherorder(theta_fit, *popt)
        r_crc = r_theta_circle(theta_fit, *popt_crc)
        r_ellipse = r_theta_ellipse(theta_fit, *popt_crc, *popt_ell)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x - cx, y - cy, 'k.', label='Data')
        ax.plot(r_crc * np.cos(theta_fit), r_crc * np.sin(theta_fit), 'b:', label=r'Circle ($\bar r=$' + '{0:.1f}px)'.format(guess_rbar))
        ax.plot(r_ellipse * np.cos(theta_fit), r_ellipse * np.sin(theta_fit), 'g-', label=r'Ellipse ($\gamma_2=$' + '{0:.1f}%)'.format(popt_ell[0]*100))
        ax.plot(r_fit * np.cos(theta_fit), r_fit * np.sin(theta_fit), 'r--', lw=2, label=r'Fit ($\Sigma\gamma_{n>2}/\gamma_2=$' + '{0:.2f})'.format(non_ellip))
        ax.set_aspect('equal')
        ax.yaxis.set_major_locator(ax.xaxis.get_major_locator())
        ax.legend()
        ax.grid(True)
        if plot_savedir is not None:
            fig.savefig(os.path.join(plot_savedir, f'fit_{frame_n}.png'))

    if print_res:
        strout = ''
        strout += '\n----------+------+-------+-------+-------+-------'
        strout += '\nParameter | unit |  raw  | guess | value | err'
        strout += '\n----------+------+-------+-------+-------+-------'
        strout += '\n    x0    |  px  |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(0, res['x0_el'], res['x0'], res['x0_er'])
        strout += '\n    y0    |  px  |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(0, res['y0_el'], res['y0'], res['y0_er'])
        strout += '\n   rbar   |  px  |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(res['rmean'], res['r0_el'], res['rbar'], res['rb_er'])
        strout += '\n  gamma_2 |  %   |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(res['rawg2']*100, res['g2_el']*100, res['g2']*100, res['g2_er']*100)
        strout += '\n  gamma_3 |  %   |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(0, 0, res['g3']*100, res['g3_er']*100)
        strout += '\n  gamma_4 |  %   |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(0, 0, res['g4']*100, res['g4_er']*100)
        strout += '\n  gamma_5 |  %   |{0:6.2f} |{1:6.2f} |{2:6.2f} | {3:4.3f}'.format(0, 0, res['g5']*100, res['g5_er']*100)
        strout += '\n----------+------+-------+-------+-------+-------'
        strout += '\n\nFine-tuned center: ({0:.3f},{1:.3f}). Guess from circle fit: ({3:.3f},{3:.3f})'.format(res['x0'], res['y0'], res['x0_el'], res['y0_el'])
        strout += '\nDroplet radius: guess={0:.2f} px, circle fit={1:.2f} px, Final fit={2:.2f} px'.format(res['rmean'], res['r0_el'], res['rbar'])
        strout += '\nElliptical deformation: guess={0:.2f}%, elliptical fit={1:.2f}%, Final fit={2:.2f}%'.format(res['rawg2']*100, res['g2_el']*100, res['g2']*100)
        strout += '\nHigher-order deformation coefficients: g3={0:.2f}%, g4={1:.2f}%, g4={2:.2f}%'.format(res['g3']*100, res['g4']*100, res['g5']*100)
        strout += '\nNon-elliptical deformation parameter: {0:.3f}'.format(res['nonel'])
        strout += '\nRMS fit error: {0:.2f} px, fit R2: {1:.3f}'.format(res['mserr'], res['R2'])
        iof.printlog(strout, flog)
            
    return res

def analyze_deformations(img_path, track_df, crop_roi_size, img_bkg=None, img_bkgcorr_offset=0, img_blur_sigma=0, 
                         edge_threshold=5.5, smoothing_iterN=4, dbscan_eps=1, dbscan_minN=1, filter_r_thr=100, 
                         allowed_badpoints=0, plot_outdir=None, px_size=1, fps=1, PID_list=None, flog=None):
    
    # Final lists to store results for successfully processed particles
    res_list, PID_sel = [], []
    if PID_list is None:
        PID_list = track_df['particle'].unique()

    count_skipped = 0
    t0 = time.time()
    for pID in PID_list:
        iof.printlog('Now processing particle {0}/{1} (PID: {2})...'.format(len(PID_sel)+count_skipped+1, len(PID_list), pID), flog)
        
        # Flag to track if the particle processing fails at any point
        particle_failed = False

        # Temporary lists for the current particle's data
        cur_res = []

        # Create directories for saving plots
        cur_outdir = None
        do_plot = False 
        if plot_outdir is not None:
            cur_outdir = os.path.join(plot_outdir, str(pID))
            os.makedirs(cur_outdir, exist_ok=True)
            do_plot = True

        count_bad = 0
        for framenum in track_df[track_df['particle'] == pID]['frame']:
            
            cur_record = track_df[(track_df['particle'] == pID) & (track_df['frame'] == framenum)]
            xloc, yloc = int(cur_record['x'].iloc[0]), int(cur_record['y'].iloc[0])
            
            # Extract Region of Interest (ROI) and process image
            drop_ROI = [xloc-crop_roi_size, yloc-crop_roi_size, xloc+crop_roi_size, yloc+crop_roi_size]
            drop_img = iof.get_single_frame(img_path, framenum, cropROI=drop_ROI, bkg=img_bkg, bkgcorr_offset=img_bkgcorr_offset, 
                                            blur_sigma=img_blur_sigma, dtype=float)
            drop_edges = find_edges(drop_img, edge_threshold=edge_threshold, smoothing_iterN=smoothing_iterN, 
                                    dbscan_eps=dbscan_eps, dbscan_minN=dbscan_minN, plot=False)
            
            # If circle extraction fails for any frame, mark the whole particle as failed
            if drop_edges is None:
                iof.printlog(f"Clustering analysis failed on frame {framenum} for particle {pID}.", flog)
                particle_failed = True
                count_bad += 1
            if drop_edges.shape[0] < 6:
                iof.printlog(f"Analysis failed on frame {framenum}: too few edge datapoints to fit particle {pID}.", flog)
                particle_failed = True
                count_bad += 1
            
            if particle_failed:
                if count_bad > allowed_badpoints:
                    iof.printlog(f"Number of bad frames ({count_bad}) exceeded limit threshold ({allowed_badpoints}): skipping particle {pID}.", flog)
                    break
            else:
                # Continue with calculations if frame analysis was successful
                fitres = fit_edge(drop_edges, filter_r_thr=filter_r_thr, frame_n=framenum, plot=do_plot, plot_savedir=cur_outdir)
                cur_res.append(fitres)
                for key in fitres:
                    track_df.loc[(track_df['particle'] == pID) & (track_df['frame'] == framenum), 'fit_'+key] = fitres[key]
        
        plt.close('all')

        # After processing all frames, check if the particle failed.
        # If it did, skip to the next particle without saving its results.
        if count_bad>allowed_badpoints:
            count_skipped += 1
            # Optional: Clean up the directory created for the failed particle
            if cur_outdir is not None:
                try:
                    shutil.rmtree(cur_outdir)
                    iof.printlog(f"Removed directory for failed particle {pID}.", flog)
                except OSError as e:
                    iof.printlog(f"Error removing directory {cur_outdir}: {e.strerror}", flog)
                continue # Skip to the next particle
        else:
            # If the particle was processed successfully, append its results to the final lists
            res_list.append(cur_res)
            PID_sel.append(pID)
            
    iof.printlog('Analysis completed in {0:.1f} seconds. {1} particles successfully analyzed, {2} skipped'.format(time.time()-t0, len(PID_sel), count_skipped), flog)

    return res_list, PID_sel

def merge_trackres(track_df_list):
    return None

def filter_droplets(track_df, thr_relstd, thr_mserr, allowed_badpoints=0, PID_list=None, save_fig=None, flog=None):
    pID_subset = []
    pID_excl = []
    
    if PID_list is None:
        PID_list = track_df['particle'].unique()
    
    if len(PID_list)>0 and 'fit_rbar' in track_df.columns:
        
        fig, ax = plt.subplots(nrows=2, figsize=(8,8))
        for pID in PID_list:
            cur_subdf = track_df[(track_df['particle'] == pID)]
            if np.std(cur_subdf['fit_rbar'])/np.mean(cur_subdf['fit_rbar']) > thr_relstd or np.max(cur_subdf['fit_mserr'] > thr_mserr):
                count_nan = 0
                if allowed_badpoints>0:
                    cur_rmean = np.mean(cur_subdf['fit_rbar'])
                    for i in range(len(cur_subdf)):
                        if np.abs(cur_subdf['fit_rbar'].iloc[i]/cur_rmean - 1) > thr_relstd or cur_subdf['fit_mserr'].iloc[i] > thr_mserr:
                            for c in ['fit_rbar', 'fit_g2', 'vx']:
                                track_df.loc[(track_df['particle'] == pID) & ('frame'==cur_subdf['frame'].iloc[i]), c] = np.nan
                            count_nan += 1
                if allowed_badpoints>0 and count_nan<=allowed_badpoints:
                    fmt = '--'
                    lw=2
                    pID_subset.append(int(pID))
                    iof.printlog('Particle {0} marked as acceptable despite {1} bad datapoints (tolerated {2} bad datapoints at most)'.format(pID, count_nan, allowed_badpoints), flog)
                else:
                    fmt = ':'
                    lw=1
                    pID_excl.append(int(pID))
            else:
                fmt = '.-'
                lw=2
                pID_subset.append(int(pID))
            ax[0].plot(cur_subdf['x'], cur_subdf['fit_rbar'], fmt, lw=lw, alpha=0.5, label=str(pID))
            ax[1].plot(cur_subdf['x'], cur_subdf['fit_mserr'], fmt, lw=lw, alpha=0.5, label=str(pID))
        ax[0].legend(ncol=10, prop={'size': 6})
        ax[1].set_yscale('log')
        ax[0].set_ylabel(r'$\bar r$ [px]')
        ax[1].set_ylabel(r'$\langle r - r_{fit} \rangle$ [px]')
        ax[1].set_xlabel(r'$x$ [px]')

        if save_fig is not None:
            fig.savefig(save_fig)

        iof.printlog('{0} excluded particles, final analysis will be fine tuned on {1} particles:\n{2}'.format(len(pID_excl), len(pID_subset), pID_subset), flog)
        
    else:
        iof.printlog('ERROR: input dataset has no valid particles', flog)            
        
    return pID_subset

def plot_lissajous(track_df, pID_subset=None, mod_df=None, drop_sort='Gp', recalc_stress=False, 
                   px_size=1, fps=1, eta=1, ss_maxtomean=None, plot_alpha=0.3, save_fig=None):
    if recalc_stress or 'stress' not in track_df.columns:
        pID_subset_stress = track_postproc(track_df, px_size=px_size, fps=fps, eta=eta, ss_maxtomean=ss_maxtomean, plot=False, verbose=0)
        if pID_subset is not None:
            pID_subset = [pID for pID in pID_subset if pID in pID_subset_stress]
            
    if pID_subset is None:
        pID_subset = track_df['particle'].unique()

    if mod_df is not None:
        if drop_sort is mod_df.columns:
            pID_subset = [x for _, x in sorted(zip(mod_df[drop_sort], pID_subset))]
    colors = plt.cm.jet(np.linspace(0,1,len(pID_subset)))
    fig, ax = plt.subplots(2,2)
    for i in range(len(pID_subset)):
        pID = pID_subset[i]
        cur_subdf = track_df[(track_df['particle'] == pID)]
        t = np.arange(0, len(cur_subdf)) / fps
        ax[0,0].plot(t, cur_subdf['stress'], 'o', color=colors[i], alpha=plot_alpha, label=str(pID))
        ax[1,1].plot(cur_subdf['fit_g2'], t, 'o', color=colors[i], alpha=plot_alpha)
        ax[0,1].plot(cur_subdf['fit_g2'], cur_subdf['stress'], 'o', color=colors[i], alpha=plot_alpha)
        if mod_df is not None:
            if len(mod_df)>0:
                mod_curp = mod_df[(mod_df['particle'] == pID)]
                if len(mod_curp)>0:
                    stress_fit = float(mod_curp['stress_off'].iloc[0]) + float(mod_curp['stress_amp'].iloc[0])*np.sin(float(mod_curp['omega'].iloc[0])*t + float(mod_curp['stress_ph'].iloc[0]))
                    strain_fit = float(mod_curp['strain_off'].iloc[0]) + float(mod_curp['strain_amp'].iloc[0])*np.sin(float(mod_curp['omega'].iloc[0])*t + float(mod_curp['strain_ph'].iloc[0]))
                    ax[0,0].plot(t, stress_fit, '--', color=colors[i])
                    ax[1,1].plot(strain_fit, t, '--', color=colors[i])
                    ax[0,1].plot(strain_fit, stress_fit, '--', color=colors[i])
                    pass
            
    ax[1,0].set_visible(False)
    ax[0,1].xaxis.set_visible(False)
    ax[0,1].yaxis.set_visible(False)
    ax[0,0].set_xlabel(r'$t$ [s]')
    ax[0,0].set_ylabel(r'$\sigma_{ext}$ [Pa]')
    ax[1,1].set_ylabel(r'$t$ [s]')
    ax[1,1].set_xlabel(r'$\gamma_2$ [-]')
    fig.legend(ncol=4, prop={'size': 6}, loc=3)
    
    if save_fig is not None:
        fig.savefig(save_fig)
    
    return pID_subset

def sin_oscill(t, omega, A, phi, yoff):
    return A * np.sin(omega * t + phi) + yoff

def guess_oscill_param(t_data, y_data, noscill=1):
    if len(t_data)>1 and len(y_data)>1:
        return  noscill * 2 * np.pi / (t_data[-1] - t_data[0]), np.max(y_data)-np.mean(y_data), 0, np.mean(y_data)
    else:
        return np.nan, np.nan, np.nan, np.nan

def fit_oscill(t_data, y_data, flog=None):
    if len(t_data)==len(y_data) and len(t_data)>3:
        initial_guess = guess_oscill_param(t_data, y_data)
        try:
            params, covariance = curve_fit(lambda t, omega, A, phi, yoff: sin_oscill(t_data, omega, A, phi, yoff), 
                                           t_data, y_data, p0=initial_guess)
        except RuntimeError:
            iof.printlog('ERROR: oscill fit did not converge. Trying to fit vertical and horizontal variables independently...', flog)
            params, covariance = initial_guess, None
        if covariance is None:
            try:
                niter = 2
                for i in range(niter):
                    par_w, cov_w = curve_fit(lambda t, omega, phi: sin_oscill(t, omega, initial_guess[1], phi, initial_guess[3]), 
                                             t_data, y_data, p0=[initial_guess[0], initial_guess[2]])
                    initial_guess = [par_w[0], initial_guess[1], par_w[1], initial_guess[3]]
                    iof.printlog('... horizontal fit {2}/{3} converged. omega={0:.3f} rad/s; phi={1:.3f} rad'.format(*par_w, i, niter), flog)
                    par_A, cov_A = curve_fit(lambda t, A, yoff: sin_oscill(t, initial_guess[0], A, initial_guess[2], yoff),
                                             t_data, y_data, p0=[initial_guess[1], initial_guess[3]])
                    initial_guess = [initial_guess[0], par_A[0], initial_guess[2], par_A[1]]
                    iof.printlog('... vertical fit {2}/{3} converged. A={0:.3f}; yoff={1:.3f}'.format(*par_A, i, niter), flog)
            except RuntimeError:
                iof.printlog('... ERROR: returning initial guess instead of fit parameters', flog)
            params, covariance = initial_guess, None
        if params[0] < 0:
            params[0] *= -1
            params[1] *= -1
            params[2] *= -1
        if params[1] < 0:
            params[1] *= -1
            params[2] += np.pi
        params[2] = params[2] % (2*np.pi)
        return params, covariance
    else:
        iof.printlog('ERROR: oscill fit requires two arrays of equal length with at least 4 datapoints. Current input length: {0} and {1}'.format(len(t_data), len(y_data)), flog)
        return None, None

# Define the oscillation function with a shared omega
def oscill_shared_omega(t_data, fit_params):
    n_nonshared_pars = 3
    n_datasets = (len(fit_params)-1) // n_nonshared_pars
    omega = fit_params[-1]  # Shared omega parameter
    result = []
    
    for i in range(n_datasets):
        A = fit_params[n_nonshared_pars * i]
        phi = fit_params[n_nonshared_pars * i + 1]
        yoff = fit_params[n_nonshared_pars * i + 2]
        cur_oscill = sin_oscill(t_data, omega, *fit_params[n_nonshared_pars*i:n_nonshared_pars*(i+1)])
        result.append(cur_oscill)
    
    # Flatten the result into a single array for fitting
    if len(result)>1:
        result = np.concatenate(result)
    else:
        result = result[0]
    return result

# Define the oscillation function with a shared omega
def oscill_shared_omega_2(t_data, A1, phi1, yoff1, A2, phi2, yoff2, omega):
    return oscill_shared_omega(t_data, [A1, phi1, yoff1, A2, phi2, yoff2, omega])

# Function to fit multiple datasets
def fit_oscill_shareomega(t, gamma, sigma, fit_margin=0, pre_fit_iter=2, param_bound=0, plot=False, flog=None):

    fitidx = slice(fit_margin, len(t)-fit_margin)
    fitt = t[fitidx]
    #iof.printlog('Length: {0}. Fit margin: {1}. Fitted length: {2}'.format(len(t), fit_margin, len(fitt)), flog)
    
    # First, fit independently strain and stress to have an initial guess for global fit
    gamma_fitp, _ = fit_oscill(fitt, gamma[fitidx])
    sigma_fitp, _ = fit_oscill(fitt, sigma[fitidx])
    if gamma_fitp is not None and sigma_fitp is not None:
        initial_guess = [*gamma_fitp[1:], *sigma_fitp[1:], 0.5*(gamma_fitp[0]+sigma_fitp[0])]
        
        y_data_flat = np.concatenate([gamma[fitidx], sigma[fitidx]])
        for i in range(pre_fit_iter):
            par_w, covariance = curve_fit(lambda fitt, omega, phi1, phi2: oscill_shared_omega_2(fitt, initial_guess[0], phi1, initial_guess[2],
                                                                                               initial_guess[3], phi2, initial_guess[5], omega), 
                                           fitt, y_data_flat, p0=[initial_guess[-1], initial_guess[1], initial_guess[4]], method='lm')
            initial_guess = [initial_guess[0], par_w[1], initial_guess[2], initial_guess[3], par_w[2], initial_guess[5], par_w[0]]
            par_A, covariance = curve_fit(lambda fitt, A1, yoff1, A2, yoff2: oscill_shared_omega_2(fitt, A1, initial_guess[1], yoff1,
                                                                                               A2, initial_guess[4], yoff2, initial_guess[6]), 
                                           fitt, y_data_flat, p0=[initial_guess[0], initial_guess[2], initial_guess[3], initial_guess[5]], method='lm')
            initial_guess = [par_A[0], initial_guess[1], par_A[1], par_A[2], initial_guess[4], par_A[3], initial_guess[6]]

        if param_bound is None:
            params, covariance = curve_fit(lambda *p: oscill_shared_omega(fitt, p), 
                                           fitt, y_data_flat, p0=initial_guess, method='lm')
        else:
            if param_bound==0:
                params, covariance = initial_guess, None
            else:
                param_bounds = ([x*(1-param_bound) for x in initial_guess], [x*(1+param_bound) for x in initial_guess])
                param_bounds[0][-1] = min(gamma_fitp[0], sigma_fitp[0])*(1-param_bound)
                param_bounds[1][-1] = max(gamma_fitp[0], sigma_fitp[0])*(1+param_bound)
                try:
                    params, covariance = curve_fit(lambda *p: oscill_shared_omega(fitt, p), 
                                                   fitt, y_data_flat, p0=initial_guess, method='trf',
                                                   bounds=param_bounds)
                except RuntimeError:
                    params, covariance = initial_guess, None

        # Extract fitted parameters
        guessp = {
            'A': initial_guess[0::3][:-1],  # Amplitudes for each dataset
            'phi': initial_guess[1::3],  # Phase for each dataset
            'yoff': initial_guess[2::3],  # Offset for each dataset
            'omega': initial_guess[-1]  # Shared omega
        }
        fitp = {
            'A': params[0::3][:-1],  # Amplitudes for each dataset
            'phi': params[1::3],  # Phase for each dataset
            'yoff': params[2::3],  # Offset for each dataset
            'omega': params[-1]  # Shared omega
        }

        if plot:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1, height_ratios=[1,4])
            ax = plt.subplot(gs[1])
            axres = plt.subplot(gs[0])
            ax2 = ax.twinx()
            axres2 = axres.twinx()
            gamma_guess = sin_oscill(fitt, omega=guessp['omega'], A=guessp['A'][0], phi=guessp['phi'][0], yoff=guessp['yoff'][0])
            sigma_guess = sin_oscill(fitt, omega=guessp['omega'], A=guessp['A'][1], phi=guessp['phi'][1], yoff=guessp['yoff'][1])
            gamma_fit = sin_oscill(fitt, omega=fitp['omega'], A=fitp['A'][0], phi=fitp['phi'][0], yoff=fitp['yoff'][0])
            sigma_fit = sin_oscill(fitt, omega=fitp['omega'], A=fitp['A'][1], phi=fitp['phi'][1], yoff=fitp['yoff'][1])
            ax.plot(t, gamma, 'ko', label=r'$\gamma$')
            ax.plot(fitt, gamma_guess, 'k:', label=r'$\gamma_{guess}$')
            ax.plot(fitt, gamma_fit, 'k-', label=r'$\gamma_{fit}$')
            ax2.plot(t, sigma, 'rs', label=r'$\sigma$')
            ax2.plot(fitt, sigma_guess, 'r:', label=r'$\sigma_{guess}$')
            ax2.plot(fitt, sigma_fit, 'r-', label=r'$\sigma_{fit}$')
            axres.plot(fitt, gamma[fitidx]-gamma_guess, 'k:', label=r'$\gamma-\gamma_{guess}$')
            axres2.plot(fitt, sigma[fitidx]-sigma_guess, 'r:', label=r'$\sigma-\sigma_{guess}$')
            axres.plot(fitt, gamma[fitidx]-gamma_fit, 'k-', label=r'$\gamma-\gamma_{fit}$')
            axres2.plot(fitt, sigma[fitidx]-sigma_fit, 'r-', label=r'$\sigma-\sigma_{fit}$')
            axres.set_xlim(ax.get_xlim())
            ax.set_xlabel(r'$t$ [s]')
            ax.set_ylabel(r'$\gamma$ [-]')
            ax2.set_ylabel(r'$\sigma$ [Pa]')
            axres.xaxis.set_visible(False)
            axres.set_ylabel(r'$\gamma-\gamma_{fit}$ [-]')
            axres2.set_ylabel(r'$\sigma-\sigma_{fit}$ [Pa]')
            hdl, lbl = ax.get_legend_handles_labels()
            hdl2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(hdl+hdl2, lbl+lbl2, ncol=2, prop={'size': 8})
            fig.tight_layout()

    else:
        return None, None, None
    
    return fitp, covariance, guessp

def calculate_modulus(gamma, sigma, fps=1, fit_margin=0, param_bound=0, pre_fit_iter=2, plot=False, flog=None):

    t = np.arange(0, len(gamma)) / fps
    non_nan_indices = np.logical_and(~np.isnan(gamma), ~np.isnan(sigma))
    #iof.printlog('Calculating droplet modulus, keeping {0}/{1} valid datapoints'.format(np.sum(non_nan_indices), len(t)), flog)
    t = t[non_nan_indices]
    gamma = np.array(gamma)[non_nan_indices]
    sigma = np.array(sigma)[non_nan_indices]
    
    fitp, covar, guessp = fit_oscill_shareomega(t, gamma, sigma, fit_margin=fit_margin, param_bound=param_bound, pre_fit_iter=pre_fit_iter, plot=plot, flog=flog)

    if fitp is not None:
    
        Gstar = fitp['A'][1] / fitp['A'][0]
        phase_diff = fitp['phi'][1] - fitp['phi'][0]
        res = {
            'G*' : Gstar,
            'delta' : phase_diff,
            'Gp' : Gstar * np.cos(phase_diff),
            'Gs' : Gstar * np.sin(phase_diff),
            'strain_off' : fitp['yoff'][0],
            'stress_off' : fitp['yoff'][1],
            'strain_amp' : fitp['A'][0],
            'stress_amp' : fitp['A'][1],
            'strain_ph' : fitp['phi'][0],
            'stress_ph' : fitp['phi'][1],
            'omega' : fitp['omega'],
            'npts' : len(t)-2*fit_margin,
        }

        return res
    else:
        return None

def calc_moduli(track_df, PID_list=None, fps=1, fit_margin=0, param_bound=0, pre_fit_iter=2, save_csv=None, plot=True, save_fig=None, flog=None):
    
    res = []
    if PID_list is None:
        PID_list = track_df['particle'].unique()
    if len(PID_list)>0:
        for pID in PID_list:
            cur_df = track_df[(track_df['particle'] == pID)]
            #iof.printlog('Now analyzing particle {0} ({1} time points available)'.format(pID, len(cur_df)), flog)
            cur_res = calculate_modulus(gamma=cur_df['fit_g2'], sigma=cur_df['stress'], 
                                        fps=fps, fit_margin=fit_margin, param_bound=param_bound, pre_fit_iter=pre_fit_iter, plot=False, flog=flog)
            if cur_res is not None:
                cur_res['rbar'] = np.nanmean(cur_df['fit_rbar'])
                cur_res['v_avg'] = np.nanmean(cur_df['vx'])
                cur_res['yavg'] = np.nanmean(cur_df['y'])
                cur_res['particle'] = pID
                res.append(cur_res)
        res_df = pd.DataFrame(res)

        if len(res_df)>0:
            Gp_avg, Gs_avg = np.mean(res_df['Gp']), np.mean(res_df['Gs'])
            r_avg, f_avg = np.mean(res_df['rbar']), np.mean(res_df['omega'])
            Gp_std, Gs_std = np.std(res_df['Gp']), np.std(res_df['Gs'])
            r_std, f_std = np.std(res_df['rbar']), np.std(res_df['omega'])

            if plot:

                strmsg = ''
                strmsg += '\n +------------------+'
                strmsg += '\n |  Moduli result:  |'
                strmsg += '\n +------------------+-----------------------+'
                strmsg += '\n |  droplet radius  :{0:6.1f} +/-{1:6.1f} px    |'.format(r_avg, r_std)
                strmsg += '\n |  frequency       :{0:6.1f} +/-{1:6.1f} rad/s |'.format(f_avg, f_std)
                strmsg += '\n |  storage modulus :{0:6.1f} +/-{1:6.1f} Pa    |'.format(Gp_avg, Gp_std)
                strmsg += '\n |  loss modulus    :{0:6.1f} +/-{1:6.1f} Pa    |'.format(Gs_avg, Gs_std)
                strmsg += '\n +------------------+-----------------------+\n'
                iof.printlog(strmsg, flog)

                fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(7,12))
                ax[0,0].plot(res_df['rbar'], res_df['strain_amp'], '^', c='tab:orange', label=r'$\bar\gamma$')
                ax[0,0].plot(res_df['rbar'], res_df['strain_off'], 'v', c='tab:purple', label=r'$\gamma_{off}$')
                ax[0,0].plot(res_df['rbar'], [np.mean(res_df['strain_amp'])]*len(res_df), ':', c='tab:orange')
                ax[0,0].plot(res_df['rbar'], [np.mean(res_df['strain_off'])]*len(res_df), ':', c='tab:purple')
                ax[0,1].plot(res_df['omega'], res_df['strain_amp'], '^', c='tab:orange')
                ax[0,1].plot(res_df['omega'], res_df['strain_off'], 'v', c='tab:purple')
                ax[0,1].plot(res_df['omega'], [np.mean(res_df['strain_amp'])]*len(res_df), ':', c='tab:orange')
                ax[0,1].plot(res_df['omega'], [np.mean(res_df['strain_off'])]*len(res_df), ':', c='tab:purple')
                ax[0,2].plot(res_df['yavg'], res_df['strain_amp'], '^', c='tab:orange')
                ax[0,2].plot(res_df['yavg'], res_df['strain_off'], 'v', c='tab:purple')
                ax[0,2].plot(res_df['yavg'], [np.mean(res_df['strain_amp'])]*len(res_df), ':', c='tab:orange')
                ax[0,2].plot(res_df['yavg'], [np.mean(res_df['strain_off'])]*len(res_df), ':', c='tab:purple')
                ax[1,0].plot(res_df['rbar'], res_df['stress_amp'], '>', c='tab:gray', label=r'$\bar\sigma$')
                #ax[1,0].plot(res_df['rbar'], res_df['stress_off'], '<', c='tab:olive', label=r'$\sigma_{off}$')
                ax[1,0].plot(res_df['rbar'], [np.mean(res_df['stress_amp'])]*len(res_df), ':', c='tab:gray')
                #ax[1,0].plot(res_df['rbar'], [np.mean(res_df['stress_off'])]*len(res_df), ':', c='tab:olive')
                ax[1,1].plot(res_df['omega'], res_df['stress_amp'], '>', c='tab:gray')
                #ax[1,1].plot(res_df['omega'], res_df['stress_off'], '<', c='tab:olive')
                ax[1,1].plot(res_df['omega'], [np.mean(res_df['stress_amp'])]*len(res_df), ':', c='tab:gray')
                #ax[1,1].plot(res_df['omega'], [np.mean(res_df['stress_off'])]*len(res_df), ':', c='tab:olive')
                ax[1,2].plot(res_df['yavg'], res_df['stress_amp'], '>', c='tab:gray')
                ax[1,2].plot(res_df['yavg'], [np.mean(res_df['stress_amp'])]*len(res_df), ':', c='tab:gray')
                ax[2,0].plot(res_df['rbar'], res_df['v_avg'], '*', c='tab:brown', label=r'$\langle v\rangle$')
                ax[2,0].plot(res_df['rbar'], [np.mean(res_df['v_avg'])]*len(res_df), ':', c='tab:brown')
                ax[2,1].plot(res_df['omega'], res_df['v_avg'], '*', c='tab:brown')
                ax[2,1].plot(res_df['omega'], [np.mean(res_df['v_avg'])]*len(res_df), ':', c='tab:brown')
                ax[2,2].plot(res_df['yavg'], res_df['v_avg'], '*', c='tab:brown')
                ax[2,2].plot(res_df['yavg'], [np.mean(res_df['v_avg'])]*len(res_df), ':', c='tab:brown')
                ax[3,0].plot(res_df['rbar'], res_df['Gp'], 'bo', label=r'$G^\prime$')
                ax[3,0].plot(res_df['rbar'], res_df['Gs'], 'gs', label=r'$G^{\prime\prime}$')
                ax[3,0].plot(res_df['rbar'], [Gp_avg]*len(res_df), 'b:')
                ax[3,0].plot(res_df['rbar'], [Gs_avg]*len(res_df), 'g:')
                ax[3,0].fill_between([np.min(res_df['rbar']), np.max(res_df['rbar'])], [Gp_avg-Gp_std]*2, [Gp_avg+Gp_std]*2, color='b', alpha=0.3)
                ax[3,0].fill_between([np.min(res_df['rbar']), np.max(res_df['rbar'])], [Gs_avg-Gs_std]*2, [Gs_avg+Gs_std]*2, color='g', alpha=0.3)
                ax[3,1].plot(res_df['omega'], res_df['Gp'], 'bo')
                ax[3,1].plot(res_df['omega'], res_df['Gs'], 'gs')
                ax[3,1].plot(res_df['omega'], [Gp_avg]*len(res_df), 'b:')
                ax[3,1].plot(res_df['omega'], [Gs_avg]*len(res_df), 'g:')
                ax[3,1].fill_between([np.min(res_df['omega']), np.max(res_df['omega'])], [Gp_avg-Gp_std]*2, [Gp_avg+Gp_std]*2, color='b', alpha=0.3)
                ax[3,1].fill_between([np.min(res_df['omega']), np.max(res_df['omega'])], [Gs_avg-Gs_std]*2, [Gs_avg+Gs_std]*2, color='g', alpha=0.3)
                ax[3,2].plot(res_df['yavg'], res_df['Gp'], 'bo')
                ax[3,2].plot(res_df['yavg'], res_df['Gs'], 'gs')
                ax[3,2].plot(res_df['yavg'], [Gp_avg]*len(res_df), 'b:')
                ax[3,2].plot(res_df['yavg'], [Gs_avg]*len(res_df), 'g:')
                ax[3,2].fill_between([np.min(res_df['yavg']), np.max(res_df['yavg'])], [Gp_avg-Gp_std]*2, [Gp_avg+Gp_std]*2, color='b', alpha=0.3)
                ax[3,2].fill_between([np.min(res_df['yavg']), np.max(res_df['yavg'])], [Gs_avg-Gs_std]*2, [Gs_avg+Gs_std]*2, color='g', alpha=0.3)
                ax[0,0].set_ylabel(r'$\bar\gamma$, $\gamma_{off}$ [-]')
                ax[1,0].set_ylabel(r'$\bar\sigma$ [Pa]')
                ax[2,0].set_ylabel(r'$\langle v \rangle$ [m/s]')
                ax[3,0].set_ylabel(r'$G^\prime$, $G^{\prime\prime}$ [Pa]')
                ax[4,0].hist(res_df['rbar'], color='c')
                ax[4,0].axvline(r_avg, color='k', ls=':')
                ax[4,1].hist(res_df['omega'], color='m')
                ax[4,1].axvline(f_avg, color='k', ls=':')
                ax[4,2].hist(res_df['yavg'], color='y')
                ax[4,2].axvline(np.mean(res_df['yavg']), color='k', ls=':')
                ax[4,0].set_xlabel(r'$\bar r$ [px]')
                ax[4,1].set_xlabel(r'$\omega$ [rad/s]')
                ax[4,2].set_xlabel(r'$\langle y \rangle$ [px]')
                ax[4,0].set_ylabel(r'PDF')
                ax[5,0].hist(res_df['Gp'], color='b')
                ax[5,0].axvline(Gp_avg, color='k', ls=':')
                ax[5,1].hist(res_df['Gs'], color='g')
                ax[5,1].axvline(Gs_avg, color='k', ls=':')
                ax[5,0].axvline(Gp_avg, color='k', ls=':')
                ax[5,0].set_ylabel(r'PDF')
                ax[5,0].set_xlabel(r'$G^\prime$ [Pa]')
                ax[5,1].set_xlabel(r'$G^{\prime\prime}$ [Pa]')
                ax[5,2].set_visible(False)
                for cax in ax[:4,0]:
                    cax.legend()
                for cax in [*ax[:,1], *ax[:,2]]:
                    cax.yaxis.set_visible(False)
                fig.tight_layout()

                if save_fig is not None:
                    fig.savefig(save_fig)

            if save_csv is not None:
                res_df.to_csv(save_csv)
        else:
            iof.printlog('ERROR: no valid droplet found for moduli analysis', flog)

        return res_df
    else:
        iof.printlog('ERROR: no droplet selected for moduli analysis', flog)
        return None