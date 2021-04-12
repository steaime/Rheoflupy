import warnings
import numpy as np
import scipy.spatial as spsp
import os
from os import walk
from os.path import basename
import re
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import sys
import copy
from PIL import Image
import cv2

# IMAGE SETTINGS
_config_Imgs = {
        'subfolder_name' : 'Video_original\\',   # Where the program searches for PIV output. intended to be inside root folder
        'fname_prefix' : '',
        'fname_ext' : '.tif',
        'subtract_bkg' : None,#'AVG_Data.tif',
        'grayscale_range' : [75, 170],#[-64, 64],#[-20, 20],      
        'idx_range' : [0, -1, 1],#[150, 290, 1],#[216, 806, 1],#       # Range of images to load [start_idx, end_idx, step]. Set end_idx=-1 to indicate last image
        'crop_ROI' : [0, 0, 0, 0],# [0, 0, 0, -20],     # ROI TO CROP [x,y,w,h]. If w is <=0, the program will take TotWidth-w instead. Same for h
        'def_imgroot' : 'C:\\Users\\steaime\\Documents\\Research\\Codes\\Rheoflu\\video_folder\\',
        'channel_edgeup' : 'TopEdge.txt', #None, # None or filename with upper channel edges. Path relative to the main root folder
        'channel_edgedwn': 'BottomEdge.txt', #None,
        }

# PIV SETTINGS
_config_PIV = {
        'subfolder_name' : 'PIV_ALL_128_64_32_24\\', #'PIV_wDrop_12x12\\',   # Where the program searches for PIV output. intended to be inside root folder
        'fname_prefix' : 'PIV',
        'fname_ext' : '.txt',
        'idx_range' : [0, -1, 1],#[150, 290, 1],#[0, -1, 1],       # Range of images to load [start_idx, end_idx, step]. Set end_idx=-1 to indicate last image
        'field_sep' : '\t',
        'hdr_len' : 0,
        'grid_shape' : None,         # PIV grid shape [n_cols, n_rows]. Set to None to extract it from PIV output files
        'stress_cols' : [],#[8],         # Eventually specify column index for relevant stress values
        }

# DROPLET FINDING SETTINGS
_config_find = {
        'template_fname' : 'template_160.raw',
        'template_imgtype' : 'raw',
        'template_shape' : [160, 160],      # [n_rows, n_cols]
        'template_datatype' : 'B',#'>f',
        'match_method' : 'cv2.TM_CCORR_NORMED',
        'threshold_corr' : 0.98,
        'img_corrected' : False, # True(False) to match template against background-corrected (not corrected) images
        'log_name' : 'matchlog.txt',
        'save_subfolder' : 'Droplet_new\\',
        'save_prefix' : '',
        'save_ext' : '.tif',
        'save_idxlen' : 5,
        'save_corrected' : True, # True|False to save in a subfolder of the analysis folder named after 'save_subfolder'
                                  # Background corrected|Original images.
                                  # Note: What is saved in the output folder is still dictated by whether 'subtract_bkg'==None or not
        'save_window' : [0, -15, 190, 190], # ROI TO SAVE [x,y,w,h]
        'save_pair' : True, # True to save couples of pictures with same ROI cut. To do PIV in the lab reference frames 
        'pair_suffix' : ['_a', '_b'],
        'find_max_num' : 1, # If >1: after finding the first maximum, it looks around (beyond the template size) for a second match larger than threshold_corr
                            # and so on until it reaches find_max_num or doesn't find anything. Set to 0 to skip location step
        'log_first_rejected' : False,
        }

# DROPLET EDGE SETTINGS
_config_edge = {
        'subfolder_name' : 'Edges_DropFrame\\', # None to skip step
        'fname_prefix' : 'EDGE_',
        'fname_ext' : '_a.txt',
        'fname_frameidx_pos' : -1, # index of the integer with frame number in filename
        'idx_range' : [0, -1, 1],
        'idx_from_file' : 'txt', # 'img'|'drop'|'none'|'ext' to associate edge file to image file:
                                 # - 'image': matching index in Edge filename with index in image filename
                                 # - 'drop' : matching index in Edge filename with droplet index 
                                 # - 'none' : matching file index in list, 1-based, with droplet index
                                 # - 'txt'  : use external text file to match index in Edge filename with index in image filename
        'idx_from_file_txt' : 'Droplet\\matchlog.txt', # if idx_from_file=='txt', path of the file (relative to the root folder)
        'idx_from_file_filecol' :   7, # if idx_from_file=='txt', column of edge file index (default for matchlog.txt: 7)
        'idx_from_file_framecol' :  1, # if idx_from_file=='txt', column of frame number (default for matchlog.txt: 1)
        'idx_from_file_posx' :  4, 
        'idx_from_file_posy' :  5, 
        'idx_from_file_posoffset' :  [0, -15], 
        'field_sep' : ',',
        'hdr_len' : 0,
        'droplet_ref_frame' : None,#[0,16], # - None if the droplet edge is defined in the main reference frame
                                     # - otherwise: [x, y], where [0, 0] is the outcome of the template matching algorithm
        'load_normals' : True, # Load normal vector to the surface. Normal vector is positive pointing out of the droplet
        'load_forces' : True, # Load surface forces from edge files as well
        'force_format' : 'nt',  # 'nt'|'xy' to define forces in normal,tangential or x,y components.
                                # 'nt' mode assumes that normals are loaded as well. 
                                # Normal component is positive pointing out of the droplet
                                # Tangential co;ponent is positive rotating CCW around the droplet
        'fit_params' : '_FitParams.txt',#None, # None or filename with fit parameters: center of mass, mean radius, strains, stresses, outlier_flags
        'fit_delimiter' : ',',
        }

# ZOOMED PIV SETTINGS
_config_PIVz = {
        'subfolder_name' : None,#'_a_drop_for_PIV\\PIV_droplet_ref_frame\\',   # Where the program searches for PIV output. intended to be inside root folder. None to skip step
        'fname_prefix' : 'PIV',
        'fname_ext' : '.txt',
        'idx_range' : [0, -1, 1],       # Range of images to load [start_idx, end_idx, step]. Set end_idx=-1 to indicate last image
        'idx_from_file' : False, # True (False) to associate PIV file to droplet (image) file using index in PIV file (using file index in list, 1-based)
        'field_sep' : '\t',
        'hdr_len' : 0,
        'grid_shape' : None,         # PIV grid shape [n_cols, n_rows]. Set to None to extract it from PIV output files
        }

# OUTPUT SETTINGS
_config_output = {
        'figname_prefix' : 'PLOT_',
        'figname_idxlen' : 5,
        'figname_ext' : '.png',
        'figtitle' : None,
        'fig_idx' : 'fig',              # 'auto'|'fig'|'PIV' to index output figure from 0-based-index|figure#|PIV#
        'hide_axes' : True,
        'folder' : 'C:\\Users\\steaime\\Documents\\Research\\Codes\\Rheoflu\\', # Set to None not to ouput anything
        'subfolder_suffix' : '_out\\',
        'save_fullframe' : True,   # Set to false to save only zoomed (droplet-centric) videos
        'zoom_subfolder' : 'Zoom\\', # None not to save zoomed videos. 
                                     # Warning: if not saving zoomed videos, combined videos won't display some features such as overlays
        'zoom_save_previous' : False, ##### WARNING: DELICATE FLAG. Set it to True if you isolated pairs of images (save_pair == True)
                                        # and did the PIV/edge finding on the first image of the pair (thus you want to plot that one
                                        # instead of the current frame in the output)
        'combined_subfolder' : 'Combined\\', # If not None, also save a 'combined' image, of original + zoom (if present)
        'combined_zoompos' : 'lower right', # Up to now only 'lower left'|'lower right'|'upper left'|'upper right' implemented
        'combined_gsopts' : {'nrows':2, 'ncols':2, 'left':0.0, 'right':1.0, 'wspace':0.0, 'hspace':0.0, 'height_ratios':[1, 0.75], 'width_ratios':[1, 0.32]},
                    # Use the following for 4:3 video output:
                    #{'nrows':2, 'ncols':2, 'left':0.0, 'right':1.0, 'wspace':0.0, 'hspace':0.0, 'height_ratios':[1, 1.2], 'width_ratios':[1, 0.73]},
        'combined_zoom_idx' : 0, # In case of multiple found droplets, index of the droplet to show in the inset
        'figsize' : (9.06, 3),
        'zoom_figsize' : (5, 5),
        'combined_figsize' : (15,9), # if None, no figure size will be specified, constrained_layout will be used instead
        'img_cmap' : 'Greys_r',
        'img_vbounds' : [60, 180],#[0, 256],  #[-64, 64],#
        'overlay_type' : 'avg_norm', # Color overlay. None to avoid overlay. 'norm' to use velocity norm, 'u|v to have x|y projections
                                 # Prefix avg_ to use time-averaged values instead of instantaneous ones
                                 # 'extra#' to use #-th extra column (typically:stress) 
        'overlay_vbounds' : None,#[0, 12.5], # If not None, set manual vmin, vmax (useful to avoid fluctuations if not using time averaged quantities)
        'overlay_cmap' : plt.cm.get_cmap('hot'),
        'overlay_alpha_pwr' : 1,
        'overlay_alpha_clip' : [0, 0.2],
        'rectangleopts' : dict(linewidth=1, edgecolor='b', facecolor='none'),
        'rect_margin' : 1,
        'edge_hull' : True, # True (False) to draw edge hull (to draw individual points)
        'edge_plot' : 'g.', # None not to plot edge
        'edge_fit_plot' : 'g-', # None not to plot edge
        'center_fit_plot' : 'kx',
        'edge_fit_npts' : 100,
        'edge_fill' : {'color':'g', 'alpha':0.2}, # None or keyword dictionary with color
        'edge_mask_PIV' : True, # True (False) to mask (not to mask) PIV data enclosed in edges
        'stream_seedstep' : None, # Seed streamlines every N PIV grid nodes. None to avoid seeding.
        'stream_color' : 'norm', # If not None, same idea as overlay...
        'stream_avg' : True, # If True, use time-averaged PIV data to generate streamlines
        'streamopts' : dict(density=[1.2,0.8], linewidth=1.0, color='r', cmap='hot', arrowsize=1.0, minlength=0.03), # If None, go for a quiver. Otherwise, draw a streamplot.
        'quiveropts' : dict(color='r', units='width', scale=0.5, scale_units='xy', headlength=3, pivot='tail', width=0.003, headwidth=0.5, alpha=0.7),
        'quiverkey' : None,#[0.7, 0.9, 0.2],  # quiverkey [pos_x, pos_y, len] see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.quiverkey.html
        'qk_label' : r'$0.2 \frac{pix}{frame}$',
        'plot_normalF' : dict(color='darkgreen', alpha=0.8, linewidth=1.0, head_width=2.0, head_length=3.0, length_includes_head=True), # None or dictionnary with keywords for pyplot.arrow()
        'plot_normalF_fit' : True,
        'plot_normalF_fit_maxorder' : 2,
        'plot_normalF_fit_smoothT' : 10, #int>=0. To have smoother results, average force fit parameters of i-th droplet over valid (non-flagged) fit parameters in range [i-T, i+T]
        'normalF_scale' : 1.0, #
        'ch_edge_line' : dict(color=(0.05, 0.05, 0.05, 0.6), linewidth=2.0),
        'ch_edge_shade' : dict(color=(1.0, 1.0, 1.0, 0.5)),
        }

# PROGRAM SETTINGS
_config_program = {
        'Verbose' : 1,
        }

# GLOBAL CONFIG
_config = {
        'PIV' : _config_PIV,
        'OUT' : _config_output,
        'IMG' : _config_Imgs,
        'Find' : _config_find,
        'Edge' : _config_edge,
        'PIVz' : _config_PIVz,
        'Prog' : _config_program
        }

   
def AllIntInStr(my_string):
    arr_str = re.findall(r'\d+', my_string)
    res_int = []
    for item in arr_str:
        res_int.append(int(item))
    return res_int

def FirstIntInStr(my_string):
    arr = AllIntInStr(my_string)
    if (len(arr) > 0):
        return arr[0]
    else:
        return None

def LastIntInStr(my_string):
    arr = AllIntInStr(my_string)
    if (len(arr) > 0):
        return arr[len(arr)-1]
    else:
        return None
    
def CheckFolderExists(folderPath):
    return os.path.isdir(folderPath)

def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        print("Created folder: {0}".format(folderPath))
        os.makedirs(folderPath)
        return False

def CheckFileExists(filePath):
    try:
        return os.path.isfile(filePath)
    except:
        return False
    
def FindFileNames(FolderPath, Prefix='', Ext='', IdxRange=None, FilterString='', ExcludeStrings=[], Verbose=0, AppendFolder=False):

    if Verbose>0:
        print('Sarching {0}{1}*{2}'.format(FolderPath, Prefix, Ext))
    
    FilenameList = []
    
    # get list of all filenames in raw difference folder
    for (dirpath, dirnames, filenames) in walk(FolderPath):
        FilenameList.extend(filenames)
        break
    
    if Verbose>0:
        print('before filter: {0} files'.format(len(FilenameList)))
    
    if (len(Prefix) > 0):
        FilenameList = [i for i in FilenameList if str(i).find(Prefix) == 0]
        
    if (len(Ext) > 0):
        FilenameList = [i for i in FilenameList if i[-len(Ext):] == Ext]
        
    if (len(FilterString) > 0):
        FilenameList = [i for i in FilenameList if FilterString in i]
    
    if (len(ExcludeStrings) > 0):
        for excl_str in ExcludeStrings:
            FilenameList = [i for i in FilenameList if excl_str not in i]
    
    if Verbose>0:
        print('after filter: {0} files'.format(len(FilenameList)))
    
    if AppendFolder:
        for i in range(len(FilenameList)):
            FilenameList[i] = FolderPath + FilenameList[i]
        
    if (IdxRange is not None):
        if (IdxRange[1] < 0):
            IdxRange[1] = len(FilenameList)
        resList = []
        for idx in range(IdxRange[0], IdxRange[1], IdxRange[2]):
            resList.append(FilenameList[idx])
        return resList
    else:
        return FilenameList

def LoadAllPIV(froot, fnames, separator=',', header=0, load_extra_cols=[]):
    
    PIV_data = []
    avg_x = []
    avg_y = []
    avg_u = []
    avg_v = []
#    num_frames = 0
    for cur_f in fnames:
        cur_x, cur_y , cur_u, cur_v, cur_extra = OpenPIVresult(froot+cur_f, separator=separator, header=header, load_extra_cols=load_extra_cols)
        if (len(cur_x) > 0):
            avg_x.append(cur_x)
            avg_y.append(cur_y)
            avg_u.append(cur_u)
            avg_v.append(cur_v)
#            num_frames += 1
#            if (len(avg_x) == 0):
#                avg_x = cur_x
#                avg_y = cur_y
#                avg_u = cur_u
#                avg_v = cur_v
#            else:
#                avg_x = np.add(avg_x, cur_x)
#                avg_y = np.add(avg_y, cur_y)
#                avg_u = np.add(avg_u, cur_u)
#                avg_v = np.add(avg_v, cur_v)
        PIV_data.append({'x':cur_x, 'y':cur_y,\
                         'u':cur_u, 'v':cur_v, 'extra':cur_extra})
#    avg_x = np.true_divide(avg_x, num_frames)
#    avg_y = np.true_divide(avg_y, num_frames)
#    avg_u = np.true_divide(avg_u, num_frames)
#    avg_v = np.true_divide(avg_v, num_frames)
    avg_x = np.nanmean(avg_x, axis=0)
    avg_y = np.nanmean(avg_y, axis=0)
    avg_u = np.nanmean(avg_u, axis=0)
    avg_v = np.nanmean(avg_v, axis=0)
    
    return PIV_data, {'x':avg_x, 'y':avg_y, 'u':avg_u, 'v':avg_v}
   

def OpenPIVresult(fname, separator=',', header=0, load_extra_cols=[]):
    cur_x = []
    cur_y = []
    cur_u = []
    cur_v = []
    if load_extra_cols is None:
        load_extra_cols = []
    extra_cols = []
    for icol in load_extra_cols:
        if icol is not None:
            extra_cols.append([])
    with open(fname) as fin:
        icount = 0
        for line in fin:
            if icount >= header:
                try:
                    words = line.split(separator)
                    cur_x.append(int(words[0]))
                    cur_y.append(int(words[1]))
                    cur_u.append(float(words[2]))
                    cur_v.append(float(words[3]))
                    for icol in range(len(load_extra_cols)):
                        if load_extra_cols[icol] is not None:
                            extra_cols[icol].append(float(words[load_extra_cols[icol]]))
                except:
                    pass
            icount += 1
    for i in range(len(extra_cols)):
        extra_cols[i] = np.asarray(extra_cols[i], dtype=float)
    return np.asarray(cur_x, dtype=int), np.asarray(cur_y, dtype=int), np.asarray(cur_u, dtype=float), np.asarray(cur_v, dtype=float), extra_cols

# Frame index: index of the integer with frame number in filename
def ReadAllEdgeFiles(froot, fnames, separator=',', header=0, readNorm=False, readForce=False, forceFormat='xy', FrameIndexPos=-2):
    listRes = []
    listNorm = []
    listForce = []
    listIdx = []
    for cur_fname in fnames:
        cur_x, cur_y, cur_nx, cur_ny, cur_Fx, cur_Fy = ReadEdgeFile(froot+cur_fname, separator=separator, header=header, readNorm=readNorm, readForce=readForce, forceFormat=forceFormat)
        #print(str(cur_fname) + ' - ' + str(len(cur_x)))
        listNorm.append([cur_nx, cur_ny])
        listForce.append([cur_Fx, cur_Fy])
        listRes.append([cur_x, cur_y])
        listIdx.append(AllIntInStr(cur_fname)[FrameIndexPos])
    sorted_Res = [x for _,x in sorted(zip(listIdx,listRes))]
    if (readNorm):
        sorted_Norm = [x for _,x in sorted(zip(listIdx,listNorm))]
    else:
        sorted_Norm = None
    if (readForce):
        sorted_Force = [x for _,x in sorted(zip(listIdx,listForce))]
    else:
        sorted_Force = None
    return sorted_Res, sorted_Norm, sorted_Force, sorted(listIdx)

def ReadEdgeFile(fname, separator=',', header=0, readNorm=False, readForce=False, forceFormat='xy'):
    cur_x = []
    cur_y = []
    cur_nx = []
    cur_ny = []
    cur_Fx = []
    cur_Fy = []
    with open(fname) as fin:
        icount = 0
        for line in fin:
            if icount >= header:
                #try:
                    words = line.split(separator)
                    cur_x.append(float(words[0]))
                    cur_y.append(float(words[1]))
                    cur_idx = 2
                    if (readNorm):
                        cur_idx += 2
                        tmp_nx, tmp_ny = float(words[2]), float(words[3])
                        cur_nx.append(tmp_nx * 1.0 / np.hypot(tmp_nx, tmp_ny))
                        cur_ny.append(tmp_ny * 1.0 / np.hypot(tmp_nx, tmp_ny))
                    if (readForce):
                        if (forceFormat == 'xy'):
                            tmp_fx, tmp_fy = float(words[cur_idx]), float(words[cur_idx+1])
                        else:
                            if (readNorm):
                                tmp_fn, tmp_ft = float(words[cur_idx]), float(words[cur_idx+1])
                                tmp_fx = tmp_fn*tmp_nx - tmp_ft*tmp_ny
                                tmp_fy = tmp_fn*tmp_ny + tmp_ft*tmp_nx
                            else:
                                tmp_fx, tmp_fy = np.nan, np.nan
                        cur_Fx.append(tmp_fx)
                        cur_Fy.append(tmp_fy)
                #except:
                #    pass
            icount += 1
    if (readForce):
        if (readNorm):
            return cur_x, cur_y, cur_nx, cur_ny, cur_Fx, cur_Fy
        else:
            return cur_x, cur_y, None, None, cur_Fx, cur_Fy
    else:
        return cur_x, cur_y, None, None, None, None

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, spsp.Delaunay):
        hull = spsp.Delaunay(hull)

    return hull.find_simplex(p)>=0











if __name__ == '__main__':
    
    if (_config['Prog']['Verbose'] > 0):
        print('\n\nPROGAM STARTED!')
    
    froot_list = []
    if (len(sys.argv) > 1):
        for i in range(1, len(sys.argv)):
            froot = sys.argv[i]
            if (froot[-1] != '/' and froot[-1] != '\\'):
                froot = froot + '\\'
            if (os.path.isdir(froot)):
                froot_list.append(froot)
            else:
                print("\nERROR reading folder root from command line: {0}".format(froot))
        if (len(froot_list)>0):
            print('\n{0} folders read from command line: {1}'.format(len(froot_list), froot_list))
    if (len(froot_list) == 0):
        froot_list = [_config['IMG']['def_imgroot']]
        print('\nUsing default folder root: {0}'.format(froot_list[0]))

    
    for froot in froot_list:
        
        if CheckFolderExists(froot):
            
            folder_title = basename(froot[:-1])
            
            if (_config['OUT']['folder'] is not None):
                OUT_folder = _config['OUT']['folder']+folder_title+_config['OUT']['subfolder_suffix']
                CheckCreateFolder(OUT_folder)
                if (_config['OUT']['zoom_subfolder'] is not None):
                    Zoom_folder = OUT_folder + _config['OUT']['zoom_subfolder']
                    CheckCreateFolder(Zoom_folder)
                else:
                    Zoom_folder = None
                if (_config['OUT']['combined_subfolder'] is not None):
                    Combined_folder = OUT_folder + _config['OUT']['combined_subfolder']
                    CheckCreateFolder(Combined_folder)
                else:
                    Combined_folder = None
            else:
                OUT_folder = None
                Zoom_folder = None
                Combined_folder = None
            
            # Search for background
            img_back = None
            if (_config['IMG']['subtract_bkg'] is not None):
                if (CheckFileExists(froot+_config['IMG']['subtract_bkg'])):
                    img_bk = Image.open(froot+_config['IMG']['subtract_bkg'])
                    img_back = np.array(img_bk, dtype=float)
            
            # Search for template
            img_template = None
            find_logfile = None
            if (_config['Find']['template_fname'] is not None):
                if (CheckFileExists(froot+_config['Find']['template_fname'])):
                    if (_config['Find']['template_imgtype'] == 'tif'):
                        img_temp = Image.open(froot+_config['Find']['template_fname'])
                        img_template = np.array(img_temp, dtype=float)
                    elif (_config['Find']['template_imgtype'] == 'raw'):
                        img_template = np.fromfile(froot+_config['Find']['template_fname'],\
                                                   dtype=_config['Find']['template_datatype'], count=-1, sep="")
                        img_template = np.reshape(img_template, _config['Find']['template_shape'])
                    print('\n{0} image template loaded from file {1} has shape {2}'.format(_config['Find']['template_imgtype'],\
                          _config['Find']['template_fname'], img_template.shape))
                    if (_config['Find']['save_subfolder'] is not None):
                        CheckCreateFolder(froot+_config['Find']['save_subfolder'])
                        if (_config['Find']['log_name'] is not None):
                            find_logfile = open(froot+_config['Find']['save_subfolder']+_config['Find']['log_name'], 'w')
                            find_logfile.write('#\tIMGidx\tDropIdx\tPIVidx\tTopLeft_x\tTopLeft_y\tMatchVal\tDropletIdx\tDropFrame_x\tDropFrame_y')
                        print('Save output in folder {0}, logfile will be {1}'.format(_config['Find']['save_subfolder'], _config['Find']['log_name']))
                else:
                    print('Warning: template image filename {0} not found in folder {1}. No template matching will be performed!'.format(_config['Find']['template_fname'], froot))
                        
            # Search for images
            if CheckFolderExists(froot+_config['IMG']['subfolder_name']):
                IMG_root = froot+_config['IMG']['subfolder_name']
                imgs_fnames = FindFileNames(IMG_root, Prefix=_config['IMG']['fname_prefix'], Ext=_config['IMG']['fname_ext'],\
                                            IdxRange=_config['IMG']['idx_range'])
                if (_config['Prog']['Verbose'] > 0):
                    print('\n{0} {1}*{2} images files found in folder {3} (index range: {4})'.format(len(imgs_fnames),\
                          _config['IMG']['fname_prefix'], _config['IMG']['fname_ext'], IMG_root, _config['IMG']['idx_range']))
            else:
                IMG_root = None
                
            # Channel edges
            chEdgeUp = None
            chEdgeDwn = None
            if (_config['IMG']['channel_edgeup'] is not None):
                if (CheckFileExists(froot+_config['IMG']['channel_edgeup'])):
                    chEdgeUp = np.loadtxt(froot+_config['IMG']['channel_edgeup'])
            if (_config['IMG']['channel_edgedwn'] is not None):
                if (CheckFileExists(froot+_config['IMG']['channel_edgedwn'])):
                    chEdgeDwn = np.loadtxt(froot+_config['IMG']['channel_edgedwn'])
            
            # Search for PIV output
            PIV_root = None
            PIV_data = None
            PIV_avg = None
            PIV_root = ''
            PIV_fnames = []
            if (_config['PIV']['subfolder_name'] is not None):
                if CheckFolderExists(froot+_config['PIV']['subfolder_name']):
                    PIV_root = froot+_config['PIV']['subfolder_name']
                    PIV_fnames = FindFileNames(PIV_root, Prefix=_config['PIV']['fname_prefix'], Ext=_config['PIV']['fname_ext'],\
                                               IdxRange=_config['PIV']['idx_range'])
                    if (_config['Prog']['Verbose'] > 0):
                        print('\n{0} {1}*{2} images files found in folder {3} (index range: {4})'.format(len(PIV_fnames),\
                              _config['PIV']['fname_prefix'], _config['PIV']['fname_ext'], PIV_root, _config['PIV']['idx_range']))
    
                    if len(PIV_fnames) > 0:
                        # Load PIV grid shape
                        if (_config['PIV']['grid_shape'] == None):
                            cur_x, cur_y , cur_u, cur_v, cur_stress = OpenPIVresult(PIV_root+PIV_fnames[0],\
                                                    separator=_config['PIV']['field_sep'], header=_config['PIV']['hdr_len'],\
                                                    load_extra_cols=_config['PIV']['stress_cols'])
                            grid_size = [len(set(cur_x)), len(set(cur_y))]
                            print('\nGrid shape {0} automatically computed from sample file {1}'.format(grid_size, PIV_root+PIV_fnames[0]))
                        else:
                            grid_size = copy.copy(_config['PIV']['grid_shape'])
                    # First process PIV output and compute average quantities (to rescale arrows and stuff like that)
                    PIV_data, PIV_avg = LoadAllPIV(PIV_root, PIV_fnames, separator=_config['PIV']['field_sep'], header=_config['PIV']['hdr_len'],\
                                                   load_extra_cols=_config['PIV']['stress_cols'])
            if (_config['OUT']['fig_idx'] == 'piv' and len(PIV_fnames) <= 0):
                raise ValueError('Indexing based on PIV frame numbers but no PIV data loaded')
                
            # Search for droplet edges
            Edge_pts = None
            Edge_idx = None
            Edge_norm = None
            Edge_forces = None
            Edge_file_idx = None
            Edge_frame_idx = None
            Edge_fitparams = None
            Edge_file_posx, Edge_file_posy = None, None
            Edge_root = ''
            Edge_fnames = []
            if (_config['Edge']['subfolder_name'] is not None):
                if CheckFolderExists(froot+_config['Edge']['subfolder_name']):
                    Edge_root = froot+_config['Edge']['subfolder_name']
                    Edge_fnames = FindFileNames(Edge_root, Prefix=_config['Edge']['fname_prefix'], Ext=_config['Edge']['fname_ext'],\
                                               IdxRange=_config['Edge']['idx_range'])
                    if (_config['Prog']['Verbose'] > 0):
                        print('\n{0} {1}*{2} edge point files found in folder {3} (index range: {4})'.format(len(Edge_fnames),\
                              _config['Edge']['fname_prefix'], _config['Edge']['fname_ext'], Edge_root, _config['Edge']['idx_range']))
                    Edge_pts, Edge_norm, Edge_forces, Edge_idx = ReadAllEdgeFiles(Edge_root, Edge_fnames, separator=_config['Edge']['field_sep'],\
                                                                       header=_config['Edge']['hdr_len'], readNorm=_config['Edge']['load_normals'], 
                                                                       readForce=_config['Edge']['load_forces'], forceFormat=_config['Edge']['force_format'],
                                                                       FrameIndexPos=_config['Edge']['fname_frameidx_pos'])
                    if (_config['Edge']['idx_from_file'] == 'txt'):
                        if CheckFileExists(froot+_config['Edge']['idx_from_file_txt']):
                            Edge_file_idx, Edge_frame_idx = np.loadtxt(froot+_config['Edge']['idx_from_file_txt'], 
                                                                       usecols=(_config['Edge']['idx_from_file_filecol'], _config['Edge']['idx_from_file_framecol']), unpack=True)
                            if (_config['Edge']['idx_from_file_posx'] is not None):
                                Edge_file_posx, Edge_file_posy = np.loadtxt(froot+_config['Edge']['idx_from_file_txt'], 
                                                                            usecols=(_config['Edge']['idx_from_file_posx'], _config['Edge']['idx_from_file_posy']), unpack=True)
                        else:
                            print('\nWARNING: Edge index file {0} not found'.format(froot+_config['Edge']['idx_from_file_txt']))
                    
                    if (_config['Edge']['fit_params'] is not None):
                        if CheckFileExists(froot+_config['Edge']['subfolder_name']+_config['Edge']['fit_params']):
                            Edge_fitparams = np.loadtxt(froot+_config['Edge']['subfolder_name']+_config['Edge']['fit_params'], delimiter=_config['Edge']['fit_delimiter'])
                else:
                    print('\nWARNING: Edge subfolder {0} not found'.format(_config['Edge']['subfolder_name']))

            # Search for zoomed PIV output
            PIVz_idx = []
            PIVz_root = None
            PIVz_data = None
            PIVz_avg = None
            PIVz_root = ''
            PIVz_fnames = []
            if (_config['PIVz']['subfolder_name']) is not None:
                if CheckFolderExists(froot+_config['PIVz']['subfolder_name']):
                    PIVz_root = froot+_config['PIVz']['subfolder_name']
                    PIVz_fnames = FindFileNames(PIVz_root, Prefix=_config['PIVz']['fname_prefix'], Ext=_config['PIVz']['fname_ext'],\
                                               IdxRange=_config['PIVz']['idx_range'])
                    if (_config['Prog']['Verbose'] > 0):
                        print('\n{0} {1}*{2} images files found in folder {3} (index range: {4})'.format(len(PIVz_fnames),\
                              _config['PIVz']['fname_prefix'], _config['PIVz']['fname_ext'], PIVz_root, _config['PIVz']['idx_range']))
    
                    if len(PIVz_fnames) > 0:
                        # Load PIV grid shape
                        if (_config['PIVz']['grid_shape'] == None):
                            cur_x_z, cur_y_z , cur_u_z, cur_v_z, dummy = OpenPIVresult(PIVz_root+PIVz_fnames[0],\
                                                    separator=_config['PIVz']['field_sep'], header=_config['PIVz']['hdr_len'])
                            grid_z_size = [len(set(cur_x_z)), len(set(cur_y_z))]
                            print('\nGrid shape {0} automatically computed from sample file {1}'.format(grid_z_size, PIVz_root+PIVz_fnames[0]))
                        else:
                            grid_z_size = copy.copy(_config['PIVz']['grid_shape'])
                        for cur_name in PIVz_fnames:
                            if _config['PIVz']['idx_from_file']:
                                PIVz_idx.append(LastIntInStr(cur_name))
                            else:
                                PIVz_idx.append(-1)
                    # First process PIV output and compute average quantities (to rescale arrows and stuff like that)
                    PIVz_data, PIVz_avg = LoadAllPIV(PIVz_root, PIVz_fnames, separator=_config['PIVz']['field_sep'], header=_config['PIVz']['hdr_len'],\
                                                     load_extra_cols=_config['PIV']['stress_cols'])
            
            
            
            
            
            
            # Build the figures
            
            FindDropletCount = 0
            prev_ROIs = None
            prev_idx = None
            for fidx in range(len(imgs_fnames)):
            
                check_idx = True
                if (PIV_root is not None and PIV_data is not None):
                    if (fidx > len(PIV_data)):
                        check_idx = False
                
                if (check_idx):
                                        
                    if (_config['OUT']['fig_idx'] == 'fig'):
                        my_idx = LastIntInStr(imgs_fnames[fidx])
                    elif (_config['OUT']['fig_idx'] == 'piv'):
                        my_idx = LastIntInStr(PIV_fnames[fidx])
                    else:
                        my_idx = fidx
                    print('{0}/{1}: working on image {2}'.format(fidx+1, len(imgs_fnames), imgs_fnames[fidx]))

                    # IMAGE
                    if (os.path.isfile(IMG_root+imgs_fnames[fidx])):
                        img = Image.open(IMG_root+imgs_fnames[fidx])
                        img_arr = np.array(img, dtype=float)
                        if (img_back is not None):
                            img_arr_corr = img_arr - img_back
                        else:
                            img_arr_corr = img_arr
                        _img_crop = _config['IMG']['crop_ROI']
                        if (_img_crop[2] <= 0):
                            _img_crop[2] = img_arr_corr.shape[1]+_img_crop[2]
                        if (_img_crop[3] <= 0):
                            _img_crop[3] = img_arr_corr.shape[0]+_img_crop[3]
                        img_arr = img_arr[_img_crop[1]:_img_crop[1]+_img_crop[3],_img_crop[0]:_img_crop[0]+_img_crop[2]]
                        img_arr_corr = img_arr_corr[_img_crop[1]:_img_crop[1]+_img_crop[3],_img_crop[0]:_img_crop[0]+_img_crop[2]]
                        if (_config['IMG']['grayscale_range'] is not None):
                            img_arr_clip = np.clip(img_arr_corr, _config['IMG']['grayscale_range'][0], _config['IMG']['grayscale_range'][1])
                        else:
                            img_arr_clip = img_arr_corr
                    else:
                        img_arr_corr = None
                        img_arr_clip = None

                    if (OUT_folder is not None):
                        if _config['OUT']['save_fullframe']:
                            if (_config['OUT']['figsize'] is None):
                                fig, ax = plt.subplots(constrained_layout=True)
                            else:
                                fig, ax = plt.subplots(figsize=_config['OUT']['figsize'])
                            if (_config['OUT']['figtitle'] is not None):
                                if (_config['OUT']['figtitle'] == True):
                                    ax.set_title(imgs_fnames[fidx])
                                else:
                                    ax.set_title(str(_config['OUT']['figtitle']))
                        else:
                            fig, ax = None, None
                        if (Combined_folder is not None):
                            if (_config['OUT']['combined_figsize'] is None):
                                figc = plt.figure(constrained_layout=True)
                            else:
                                figc = plt.figure(figsize=_config['OUT']['combined_figsize'], constrained_layout=True)
                            gs = figc.add_gridspec(**_config['OUT']['combined_gsopts'])
                            if (_config['OUT']['combined_zoompos']=='lower left'):
                                figc_axm = figc.add_subplot(gs[0, :])
                                figc_axz = figc.add_subplot(gs[1, 0])
                            elif (_config['OUT']['combined_zoompos']=='lower right'):
                                figc_axm = figc.add_subplot(gs[0, :])
                                figc_axz = figc.add_subplot(gs[1, -1])
                            elif (_config['OUT']['combined_zoompos']=='upper left'):
                                figc_axm = figc.add_subplot(gs[1, :])
                                figc_axz = figc.add_subplot(gs[0, 0])
                            elif (_config['OUT']['combined_zoompos']=='upper right'):
                                figc_axm = figc.add_subplot(gs[1, :])
                                figc_axz = figc.add_subplot(gs[0, -1])
                            else:
                                print('WARNING: combined layout ' + str(_config['OUT']['combined_zoompos']) +\
                                      ' not recognized. No combined figure will be saved')
                                figc, figc_axm, figc_axz = None, None, None
                            if (figc is not None and _config['OUT']['figtitle'] is not None):
                                if (_config['OUT']['figtitle'] == True):
                                    figc.set_title(imgs_fnames[fidx])
                                else:
                                    figc.set_title(str(_config['OUT']['figtitle']))
                        else:
                            figc, figc_axm, figc_axz = None, None, None
                    else:
                        fig, ax = None, None
                        figc, figc_axm, figc_axz = None, None, None
                    figc_axz_empty = True
                    figz_list, axz_list = [], []
                    
                    if img_arr_clip is not None:
                        if _config['OUT']['img_vbounds'] is not None:
                            if (ax is not None):
                                ax.imshow(img_arr_clip, cmap=_config['OUT']['img_cmap'], vmin=_config['OUT']['img_vbounds'][0], vmax=_config['OUT']['img_vbounds'][1])
                            if (figc_axm is not None):
                                figc_axm.imshow(img_arr_clip, cmap=_config['OUT']['img_cmap'], vmin=_config['OUT']['img_vbounds'][0], vmax=_config['OUT']['img_vbounds'][1], aspect='equal')
                        else:
                            if (ax is not None):
                                ax.imshow(img_arr_clip, cmap=_config['OUT']['img_cmap'])
                            if (figc_axm is not None):
                                figc_axm.imshow(img_arr_clip, cmap=_config['OUT']['img_cmap'], aspect='equal')
                            
                    # FIND DROPLET
                    save_ROIs = []
                    global_drop_idx = []
                    match_top_left_list = []
                    match_val_list = []
                    to_be_saved, to_be_saved_prev = [], []
                    to_be_saved_out, to_be_saved_out_prev = [], []
                    cur_droplet_idx = -1
                    if (img_template is not None and _config['Find']['find_max_num']>0):
                        if (_config['Find']['img_corrected']):
                            find_in_img = img_arr_corr
                        else:
                            find_in_img = img_arr
                        match_res = cv2.matchTemplate(find_in_img.astype(np.float32), img_template.astype(np.float32), eval(_config['Find']['match_method']))
                        
                        def EvalCorrMatrix(_matrix, _method):
                            match_min, match_max, match_min_loc, match_max_loc = cv2.minMaxLoc(_matrix)
                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            if _method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                                match_top_left = match_min_loc
                                match_val = match_min
                                match_good = (match_val < _config['Find']['threshold_corr'])
                            else:
                                match_top_left = match_max_loc
                                match_val = match_max
                                match_good = (match_val > _config['Find']['threshold_corr'])
                            return match_top_left, match_val, match_good
                        
                        match_top_left, match_val, match_good = EvalCorrMatrix(match_res, eval(_config['Find']['match_method']))
                        
                        cur_frame_found_num_drops = 0
                        while (match_good):
                            
                            cur_frame_found_num_drops += 1
                            
                            match_top_left_list.append(match_top_left)
                            match_val_list.append(match_val)
                            
                            cur_figz, cur_axz = None, None
                            
                            match_bottom_right = (match_top_left[0] + img_template.shape[1], match_top_left[1] + img_template.shape[0])
                            if (_config['OUT']['rectangleopts'] is not None):
                                rect_xy = (max(_config['OUT']['rect_margin'],match_top_left[0]),max(_config['OUT']['rect_margin'],match_top_left[1]))
                                rect_w = min(img_template.shape[1], img_arr_clip.shape[1]-match_top_left[0]-_config['OUT']['rect_margin'])
                                rect_h = min(img_template.shape[0], img_arr_clip.shape[0]-match_top_left[1]-_config['OUT']['rect_margin'])
                                if (ax is not None):
                                    match_rect_ff = patches.Rectangle(rect_xy, rect_w, rect_h, **_config['OUT']['rectangleopts'])
                                    ax.add_patch(match_rect_ff)
                                if (figc_axm is not None):
                                    match_rect_combined = patches.Rectangle(rect_xy, rect_w, rect_h, **_config['OUT']['rectangleopts'])
                                    figc_axm.add_patch(match_rect_combined)
                            if (_config['Find']['save_subfolder'] is not None):
                                save_ROIs.append([match_top_left[1]+_config['Find']['save_window'][1], match_top_left[1]+_config['Find']['save_window'][1]+_config['Find']['save_window'][3],\
                                                  match_top_left[0]+_config['Find']['save_window'][0], match_top_left[0]+_config['Find']['save_window'][0]+_config['Find']['save_window'][2]])
                                if (save_ROIs[-1][1] > img_arr_corr.shape[0]):
                                    save_ROIs[-1][0] -= save_ROIs[-1][1]-img_arr_corr.shape[0]
                                    save_ROIs[-1][1] -= save_ROIs[-1][1]-img_arr_corr.shape[0]
                                elif (save_ROIs[-1][0] < 0):
                                    save_ROIs[-1][0] -= save_ROIs[-1][0]
                                    save_ROIs[-1][1] -= save_ROIs[-1][0]
                                if (save_ROIs[-1][3] > img_arr_corr.shape[1]):
                                    save_ROIs[-1][2] -= save_ROIs[-1][3]-img_arr_corr.shape[1]
                                    save_ROIs[-1][3] -= save_ROIs[-1][3]-img_arr_corr.shape[1]
                                elif (save_ROIs[-1][2] < 0):
                                    save_ROIs[-1][2] -= save_ROIs[-1][2]
                                    save_ROIs[-1][3] -= save_ROIs[-1][2]
                                if (_config['Find']['save_corrected']):
                                    if (_config['IMG']['grayscale_range'] is not None):
                                        to_be_saved.append(np.multiply(np.subtract(np.clip(img_arr_corr[save_ROIs[-1][0]:save_ROIs[-1][1],\
                                                           save_ROIs[-1][2]:save_ROIs[-1][3]], _config['IMG']['grayscale_range'][0],\
                                                           _config['IMG']['grayscale_range'][1]), _config['IMG']['grayscale_range'][0]),\
                                                            255.0/(_config['IMG']['grayscale_range'][1]-_config['IMG']['grayscale_range'][0])).astype(np.uint8))
                                    else:
                                        to_be_saved.append(img_arr_corr[save_ROIs[-1][0]:save_ROIs[-1][1],save_ROIs[-1][2]:save_ROIs[-1][3]].astype(np.uint8))
                                else:
                                    to_be_saved.append(np.asarray(img_arr[save_ROIs[-1][0]:save_ROIs[-1][1], save_ROIs[-1][2]:save_ROIs[-1][3]], dtype=np.uint8))
                                to_be_saved_out.append(img_arr_corr[save_ROIs[-1][0]:save_ROIs[-1][1],save_ROIs[-1][2]:save_ROIs[-1][3]])
                                str_fname = froot+_config['Find']['save_subfolder']+_config['Find']['save_prefix'] + str(my_idx).zfill(_config['Find']['save_idxlen'])+\
                                                '_'+ str(len(save_ROIs)-1).zfill(2) + _config['Find']['pair_suffix'][0]+_config['Find']['save_ext']
                                imageio.imwrite(str_fname, to_be_saved[-1])
                            if (Zoom_folder is not None):
                                if (_config['OUT']['zoom_figsize'] is None):
                                    cur_figz, cur_axz = plt.subplots(constrained_layout=True)
                                else:
                                    cur_figz, cur_axz = plt.subplots(figsize=_config['OUT']['zoom_figsize'])
                                if (_config['OUT']['figtitle'] is not None):
                                    if (_config['OUT']['figtitle'] == True):
                                        cur_axz.set_title(imgs_fnames[fidx])
                                    else:
                                        cur_axz.set_title(str(_config['OUT']['figtitle']))
                                figz_list.append(cur_figz)
                                axz_list.append(cur_axz)
                            FindDropletCount += 1
                            global_drop_idx.append(FindDropletCount)
                            
                            #Erase current region in correlation map and reevaluate maximum 
                            if eval(_config['Find']['match_method']) in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                                corrmap_sub_val = np.inf
                            else:
                                corrmap_sub_val = -np.inf

                            clear_roi = [max(0, match_top_left[1]-img_template.shape[0]), min(len(match_res), match_top_left[1]+img_template.shape[0]),\
                                      max(0, match_top_left[0]-img_template.shape[1]), min(len(match_res[0]), match_top_left[0]+img_template.shape[1])]
                            
                            match_res[clear_roi[0]:clear_roi[1], clear_roi[2]:clear_roi[3]] = corrmap_sub_val
                            match_top_left, match_val, match_good = EvalCorrMatrix(match_res, eval(_config['Find']['match_method']))
                            
                            #Loop to find eventual other droplets unless we already found enough droplets
                            if (len(save_ROIs) >= _config['Find']['find_max_num']):
                                match_good = False
                            
                        #print('         {0} droplets found in frame ({1})'.format(len(global_drop_idx), global_drop_idx))
                            
                            
                            
                        if (_config['OUT']['zoom_save_previous'] or (_config['Find']['save_subfolder'] is not None and _config['Find']['save_pair'])) and prev_ROIs is not None and prev_idx is not None:
                            if (len(prev_ROIs) > 0):
                                to_be_saved_prev = []
                                for drop_frame_idx in range(len(prev_ROIs)):
                                    if (_config['Find']['save_corrected']):
                                        if (_config['IMG']['grayscale_range'] is not None):
                                            to_be_saved_prev.append(np.multiply(np.subtract(np.clip(img_arr_corr[prev_ROIs[drop_frame_idx][0]:prev_ROIs[drop_frame_idx][1],\
                                                           prev_ROIs[drop_frame_idx][2]:prev_ROIs[drop_frame_idx][3]], _config['IMG']['grayscale_range'][0],\
                                                           _config['IMG']['grayscale_range'][1]), _config['IMG']['grayscale_range'][0]),\
                                                            255.0/(_config['IMG']['grayscale_range'][1]-_config['IMG']['grayscale_range'][0])).astype(np.uint8))
                                        else:
                                            to_be_saved_prev.append(img_arr_corr[prev_ROIs[drop_frame_idx][0]:prev_ROIs[drop_frame_idx][1],\
                                                                                 prev_ROIs[drop_frame_idx][2]:prev_ROIs[drop_frame_idx][3]].astype(np.uint8))
                                    else:
                                        to_be_saved_prev.append(np.asarray(img_arr[prev_ROIs[drop_frame_idx][0]:prev_ROIs[drop_frame_idx][1],\
                                                                                   prev_ROIs[drop_frame_idx][2]:prev_ROIs[drop_frame_idx][3]], dtype=np.uint8))
                                for drop_frame_idx in range(len(to_be_saved_prev)):
                                    imageio.imwrite(froot+_config['Find']['save_subfolder']+_config['Find']['save_prefix'] + str(prev_idx).zfill(_config['Find']['save_idxlen'])+\
                                                '_' + str(drop_frame_idx).zfill(2) + _config['Find']['pair_suffix'][1]+_config['Find']['save_ext'], to_be_saved_prev[drop_frame_idx])
                        for figz_idx in range(len(figz_list)):
                            img_plotz = None
                            if _config['OUT']['zoom_save_previous']:
                                if (len(to_be_saved_out_prev)>0):
                                    img_plotz = to_be_saved_out_prev[figz_idx]
                                #print('plot previous')
                            else:
                                if (len(to_be_saved_out)>0):    
                                    img_plotz = to_be_saved_out[figz_idx]
                                #print('plot current')
                            if img_plotz is not None:
                                #print('plotz is not none')
                                if _config['OUT']['img_vbounds'] is not None:
                                    axz_list[figz_idx].imshow(img_plotz, cmap=_config['OUT']['img_cmap'], vmin=_config['OUT']['img_vbounds'][0], vmax=_config['OUT']['img_vbounds'][1])
                                else:
                                    axz_list[figz_idx].imshow(img_plotz, cmap=_config['OUT']['img_cmap'])
                                if (figc_axz is not None and figz_idx == _config['OUT']['combined_zoom_idx']):
                                    if _config['OUT']['img_vbounds'] is not None:
                                        figc_axz.imshow(img_plotz, cmap=_config['OUT']['img_cmap'], vmin=_config['OUT']['img_vbounds'][0], vmax=_config['OUT']['img_vbounds'][1], aspect='equal')
                                    else:
                                        figc_axz.imshow(img_plotz, cmap=_config['OUT']['img_cmap'], aspect='equal')
                                    figc_axz_empty = False
                                    
                        if (cur_frame_found_num_drops>0 and (_config['OUT']['zoom_save_previous'] or _config['Find']['save_subfolder'] is not None)):
                            prev_ROIs = save_ROIs
                            prev_idx = my_idx
                        else:
                            prev_ROI = []
                            prev_idx = None
                        if (find_logfile is not None):
                            for drop_frame_idx in range(len(save_ROIs)):
                                find_logfile.write('\n' + str(fidx) + '\t' + str(LastIntInStr(imgs_fnames[fidx])) + '\t' + str(drop_frame_idx) + '\t')
                                if (len(PIV_fnames) > fidx):
                                    find_logfile.write(str(LastIntInStr(PIV_fnames[fidx])))
                                else:
                                    find_logfile.write('-')
                                find_logfile.write('\t' + str(match_top_left_list[drop_frame_idx][0]) + '\t' +\
                                                   str(match_top_left_list[drop_frame_idx][1]) + '\t' + str(match_val_list[drop_frame_idx]) + '\t' +\
                                                   str(global_drop_idx[drop_frame_idx]) + '\t' + str(save_ROIs[drop_frame_idx][2]) + '\t' + str(save_ROIs[drop_frame_idx][0]))
                            if (_config['Find']['log_first_rejected']):
                                find_logfile.write('\n' + str(fidx) + '\t' + str(LastIntInStr(imgs_fnames[fidx])) + '\t-1\t-\t' +\
                                                   str(match_top_left[0]) + '\t' + str(match_top_left[1]) + '\t' + str(match_val) + '\t-1\t-\t-')
                            find_logfile.flush()
                    
                    # EDGE AND NORMAL FORCES
                    if (Edge_idx is not None and (_config['OUT']['edge_mask_PIV'] or _config['OUT']['edge_plot'] is not None or (_config['OUT']['plot_normalF'] is not None and Edge_forces is not None)) and len(save_ROIs)>0):
                        for dridx in range(len(global_drop_idx)):
                            cur_edge_idx = None
                            cur_tmp_idx = None
                            if (_config['Edge']['idx_from_file'] == 'drop'):
                                if (global_drop_idx[dridx] in Edge_idx):
                                    cur_edge_idx = Edge_idx.index(global_drop_idx[dridx])
                            elif (_config['Edge']['idx_from_file'] == 'img'):
                                if (LastIntInStr(imgs_fnames[fidx]) in Edge_idx):
                                    cur_edge_idx = Edge_idx.index(LastIntInStr(imgs_fnames[fidx]))
                            elif (_config['Edge']['idx_from_file'] == 'txt'):
                                if (LastIntInStr(imgs_fnames[fidx]) in Edge_frame_idx):
                                    cur_tmp_idx = int(np.where(Edge_frame_idx == LastIntInStr(imgs_fnames[fidx]))[0])
                                    if (Edge_file_idx[cur_tmp_idx] in Edge_idx):
                                        cur_edge_idx = Edge_idx.index(Edge_file_idx[cur_tmp_idx])
                            else:
                                if (global_drop_idx[dridx] <= len(Edge_pts)):
                                    cur_edge_idx = global_drop_idx[dridx]-1
                            if (cur_edge_idx is None):
                                
                                print('         WARNING: no edge found for {0}th droplet ({1})'.format(dridx, global_drop_idx[dridx]))
                                
                            else:
                                
                                cur_edge = Edge_pts[cur_edge_idx]
                                if (_config['Edge']['droplet_ref_frame'] is None):
                                    edge_subtract = [save_ROIs[dridx][2], save_ROIs[dridx][0]]
                                else:
                                    edge_subtract = _config['Edge']['droplet_ref_frame']
                                edge_add_shaped = np.asarray([np.ones_like(cur_edge[0])*(save_ROIs[dridx][2]-edge_subtract[0]),\
                                                                   np.ones_like(cur_edge[1])*(save_ROIs[dridx][0]-edge_subtract[1])])
                                cur_edge = np.asarray(np.add(cur_edge, edge_add_shaped))
                                cur_edge_trans = cur_edge.T
                                cur_edge_hull = spsp.ConvexHull(cur_edge_trans)
                                if (_config['OUT']['edge_hull']):
                                    cur_edge = np.asarray(cur_edge_trans[cur_edge_hull.vertices,:]).T
                                if (_config['OUT']['edge_mask_PIV'] and PIV_data is not None):
                                    if (fidx < len(PIV_data)):
                                        cur_PIV_gridpoints = np.asarray([PIV_data[fidx]['x'],PIV_data[fidx]['y']]).T
                                        cur_ma_mask = np.ma.make_mask(in_hull(cur_PIV_gridpoints, cur_edge_trans[cur_edge_hull.vertices,:]))
                                        if (isinstance(PIV_data[fidx]['u'], np.ma.MaskedArray)):
                                            PIV_data[fidx]['u'] = np.ma.masked_where(cur_ma_mask, PIV_data[fidx]['u'])
                                            #PIV_data[fidx]['u'] = np.ma.MaskedArray(PIV_data[fidx]['u'], np.ma.mask_or(cur_ma_mask, np.ma.getmask(PIV_data[fidx]['u'])))
                                        else:
                                            PIV_data[fidx]['u'] = np.ma.MaskedArray(PIV_data[fidx]['u'], cur_ma_mask)
                                shifted_edge = [np.subtract(cur_edge[0], save_ROIs[dridx][2]), np.subtract(cur_edge[1], save_ROIs[dridx][0])]
                                if Edge_fitparams is not None:
                                    if (global_drop_idx[dridx] in Edge_fitparams[:,7]):
                                        fitp_idx = int(np.where(Edge_fitparams[:,7]==global_drop_idx[dridx])[0])
                                else:
                                    fitp_idx = None
                                
                                print('         {0}th droplet ({1}) linked to edge {2} (fit params: {3})'.format(dridx, global_drop_idx[dridx], cur_edge_idx, fitp_idx))
                                
                                if fitp_idx is not None:
                                    if cur_tmp_idx is not None and Edge_file_posx is not None:
                                        cur_findx = Edge_file_posx[cur_tmp_idx] + _config['Edge']['idx_from_file_posoffset'][0]
                                        cur_findy = Edge_file_posy[cur_tmp_idx] + _config['Edge']['idx_from_file_posoffset'][1]
                                    else:
                                        cur_findx = save_ROIs[dridx][2]
                                        cur_findy = save_ROIs[dridx][0]
                                    fit_theta = np.linspace(0, 2*np.pi, _config['OUT']['edge_fit_npts'], endpoint=True)
                                    cur_rfit = Edge_fitparams[fitp_idx, 2] * (1.0 + Edge_fitparams[fitp_idx, 3] * np.cos(2*fit_theta) + Edge_fitparams[fitp_idx, 4] * np.cos(3*fit_theta))
                                    cur_edge_fit = np.asarray([np.add(Edge_fitparams[fitp_idx, 0]+cur_findx, np.multiply(cur_rfit, np.cos(fit_theta))), 
                                                               np.add(Edge_fitparams[fitp_idx, 1]+cur_findy, np.multiply(cur_rfit, np.sin(fit_theta)))])
                                    shifted_edge_fit = [np.subtract(cur_edge_fit[0], save_ROIs[dridx][2]), np.subtract(cur_edge_fit[1], save_ROIs[dridx][0])]
                                    cur_ctr_fit = [Edge_fitparams[fitp_idx, 0]+cur_findx, Edge_fitparams[fitp_idx, 1]+cur_findy]
                                    shifted_ctr_fit = [cur_ctr_fit[0]-save_ROIs[dridx][2], cur_ctr_fit[1]-save_ROIs[dridx][0]]
                                    cur_edge_fill = cur_edge_fit
                                    shifted_edge_fill = shifted_edge_fit
                                else:
                                    cur_edge_fit = None
                                    shifted_edge_fit = None
                                    cur_edge_fill = cur_edge
                                    shifted_edge_fill = shifted_edge
                                
                                if (_config['OUT']['edge_plot'] is not None):
                                    if (ax is not None):
                                        ax.plot(cur_edge[0], cur_edge[1], _config['OUT']['edge_plot'])
                                        if (cur_edge_fit is not None and _config['OUT']['edge_fit_plot'] is not None):
                                            ax.plot(cur_edge_fit[0], cur_edge_fit[1], _config['OUT']['edge_fit_plot'])
                                        if (cur_ctr_fit is not None and _config['OUT']['center_fit_plot'] is not None):
                                            ax.plot([cur_ctr_fit[0]], [cur_ctr_fit[1]], _config['OUT']['center_fit_plot'])
                                        if (_config['OUT']['edge_fill'] is not None):
                                            ax.fill(cur_edge_fill[0], cur_edge_fill[1], **_config['OUT']['edge_fill'])
                                    if (figc_axm is not None):
                                        figc_axm.plot(cur_edge[0], cur_edge[1], _config['OUT']['edge_plot'])
                                        if (cur_edge_fit is not None and _config['OUT']['edge_fit_plot'] is not None):
                                            figc_axm.plot(cur_edge_fit[0], cur_edge_fit[1], _config['OUT']['edge_fit_plot'])
                                        if (cur_ctr_fit is not None and _config['OUT']['center_fit_plot'] is not None):
                                            figc_axm.plot([cur_ctr_fit[0]], [cur_ctr_fit[1]], _config['OUT']['center_fit_plot'])
                                        if (_config['OUT']['edge_fill'] is not None):
                                            figc_axm.fill(cur_edge_fill[0], cur_edge_fill[1], **_config['OUT']['edge_fill'])
                                    if (dridx < len(figz_list)):
                                        axz_list[dridx].plot(shifted_edge[0], shifted_edge[1], _config['OUT']['edge_plot'])
                                        if (shifted_edge_fit is not None and _config['OUT']['edge_fit_plot'] is not None):
                                            axz_list[dridx].plot(shifted_edge_fit[0], shifted_edge_fit[1], _config['OUT']['edge_fit_plot'])
                                        if (shifted_ctr_fit is not None and _config['OUT']['center_fit_plot'] is not None):
                                            axz_list[dridx].plot([shifted_ctr_fit[0]], [shifted_ctr_fit[1]], _config['OUT']['center_fit_plot'])
                                        if (_config['OUT']['edge_fill'] is not None):
                                            axz_list[dridx].fill(shifted_edge_fill[0], shifted_edge_fill[1], **_config['OUT']['edge_fill'])
                                    if (figc_axz is not None and dridx == _config['OUT']['combined_zoom_idx']):
                                        figc_axz.plot(shifted_edge[0], shifted_edge[1], _config['OUT']['edge_plot'])
                                        if (shifted_edge_fit is not None and _config['OUT']['edge_fit_plot'] is not None):
                                            figc_axz.plot(shifted_edge_fit[0], shifted_edge_fit[1], _config['OUT']['edge_fit_plot'])
                                        if (shifted_ctr_fit is not None and _config['OUT']['center_fit_plot'] is not None):
                                            figc_axz.plot([shifted_ctr_fit[0]], [shifted_ctr_fit[1]], _config['OUT']['center_fit_plot'])
                                        if (_config['OUT']['edge_fill'] is not None):
                                            figc_axz.fill(shifted_edge_fill[0], shifted_edge_fill[1], **_config['OUT']['edge_fill'])
                                
                                if (_config['OUT']['plot_normalF'] is not None and Edge_forces is not None):

                                    if (_config['OUT']['plot_normalF_fit'] and fitp_idx is not None):
                                        if (_config['OUT']['plot_normalF_fit_smoothT'] <= 0):
                                            cur_sigma2 = Edge_fitparams[fitp_idx, 5]
                                            cur_sigma3 = Edge_fitparams[fitp_idx, 6]
                                        else:
                                            cur_sigma2 = np.mean([Edge_fitparams[i, 5] for i in range(max(0, fitp_idx-_config['OUT']['plot_normalF_fit_smoothT']),
                                                                                                  min(Edge_fitparams.shape[0], fitp_idx+_config['OUT']['plot_normalF_fit_smoothT']))\
                                                                                   if Edge_fitparams[i, 9]==0])
                                            cur_sigma3 = np.mean([Edge_fitparams[i, 6] for i in range(max(0, fitp_idx-_config['OUT']['plot_normalF_fit_smoothT']),
                                                                                                  min(Edge_fitparams.shape[0], fitp_idx+_config['OUT']['plot_normalF_fit_smoothT']))\
                                                                                   if Edge_fitparams[i, 9]==0])
                                        if (_config['OUT']['plot_normalF_fit_maxorder'] > 2):
                                            cur_fmag = cur_sigma2 * np.cos(2*fit_theta) + cur_sigma3 * np.cos(3*fit_theta)
                                        else:
                                            cur_fmag = cur_sigma2 * np.cos(2*fit_theta)
                                            
                                        for ptidx in range(len(cur_fmag)):
                                            cur_normalf = [cur_fmag[ptidx]*np.cos(fit_theta[ptidx]), cur_fmag[ptidx]*np.sin(fit_theta[ptidx])]
                                            
                                            if (ax is not None):
                                                ax.arrow(cur_edge_fit[0][ptidx], cur_edge_fit[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                                            if (figc_axm is not None):
                                                figc_axm.arrow(cur_edge_fit[0][ptidx], cur_edge_fit[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                                            if (figc_axz is not None and dridx == _config['OUT']['combined_zoom_idx']):
                                                figc_axz.arrow(shifted_edge_fit[0][ptidx], shifted_edge_fit[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                                                
                                    else:
                                        for ptidx in range(len(shifted_edge[0])):
                                            cur_normalf = [Edge_forces[cur_edge_idx][0][ptidx]*_config['OUT']['normalF_scale'],\
                                                                     Edge_forces[cur_edge_idx][1][ptidx]*_config['OUT']['normalF_scale']]
                                            if (ax is not None):
                                                ax.arrow(cur_edge[0][ptidx], cur_edge[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                                            if (figc_axm is not None):
                                                figc_axm.arrow(cur_edge[0][ptidx], cur_edge[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                                            if (figc_axz is not None and dridx == _config['OUT']['combined_zoom_idx']):
                                                figc_axz.arrow(shifted_edge[0][ptidx], shifted_edge[1][ptidx], cur_normalf[0], cur_normalf[1], **_config['OUT']['plot_normalF'])
                    
                    # OVERLAY
                    overlay_plot = None
                    overlay_z_plots = []
                    if (_config['OUT']['overlay_type'] == 'avg_norm' and PIV_avg is not None):
                        #overlay_plot = np.transpose(np.hypot(PIV_avg['u'], PIV_avg['v']).reshape(grid_size))
                        overlay_plot = np.hypot(PIV_avg['u'], PIV_avg['v'])
                        if (fidx < len(PIV_data)):
                            if (isinstance(PIV_data[fidx]['u'], np.ma.MaskedArray)):
                                overlay_plot = np.ma.MaskedArray(overlay_plot, np.ma.getmask(PIV_data[fidx]['u']))
                        overlay_plot = overlay_plot.reshape(grid_size).T
                        if (PIVz_avg is not None):
                            for dridx in range(len(figz_list)):
                                overlay_z_plots.append(np.transpose(np.hypot(PIVz_avg['u'], PIVz_avg['v']).reshape(grid_z_size)))
                    elif (_config['OUT']['overlay_type'] == 'avg_u' and PIV_avg is not None):
                        overlay_plot = np.transpose(np.asarray(PIV_avg['u']).reshape(grid_size))
                        if (PIVz_avg is not None):
                            for dridx in range(len(figz_list)):
                                overlay_z_plots.append(np.transpose(np.asarray(PIVz_avg['u']).reshape(grid_z_size)))
                    elif (_config['OUT']['overlay_type'] == 'avg_v' and PIV_avg is not None):
                        overlay_plot = np.transpose(np.asarray(PIV_avg['v']).reshape(grid_size))
                        if (PIVz_avg is not None):
                            for dridx in range(len(figz_list)):
                                overlay_z_plots.append(np.transpose(np.asarray(PIVz_avg['v']).reshape(grid_z_size)))
                    elif (_config['OUT']['overlay_type'] == 'norm' and PIV_data is not None):
                        if (fidx < len(PIV_data)):
                            overlay_plot = np.transpose(np.hypot(PIV_data[fidx]['u'], PIV_data[fidx]['v']).reshape(grid_size))
                        if (PIVz_data is not None and len(global_drop_idx)>0):
                            for dridx in range(len(figz_list)):
                                if (global_drop_idx[dridx] in PIVz_idx):
                                    overlay_z_plots.append(np.transpose(np.hypot(PIVz_data[PIVz_idx.index(global_drop_idx[dridx])]['u'],\
                                                            PIVz_data[PIVz_idx.index(global_drop_idx[dridx])]['v']).reshape(grid_z_size)))
                    elif (_config['OUT']['overlay_type'] == 'u'):
                        if (fidx < len(PIV_data)):
                            overlay_plot = np.transpose(np.asarray(PIV_data[fidx]['u']).reshape(grid_size))
                        if (PIVz_data is not None and len(global_drop_idx)>0):
                            for dridx in range(len(figz_list)):
                                if (global_drop_idx[dridx] in PIVz_idx):
                                    overlay_z_plots.append(np.transpose(np.asarray(PIVz_data[PIVz_idx.index(global_drop_idx[dridx])]['u']).reshape(grid_z_size)))
                    elif (_config['OUT']['overlay_type'] == 'v'):
                        if (fidx < len(PIV_data)):
                            overlay_plot = np.transpose(np.asarray(PIV_data[fidx]['v']).reshape(grid_size))
                        if (PIVz_data is not None and len(global_drop_idx)>0):
                            for dridx in range(len(figz_list)):
                                if (global_drop_idx[dridx] in PIVz_idx):
                                    overlay_z_plots.append(np.transpose(np.asarray(PIVz_data[PIVz_idx.index(global_drop_idx[dridx])]['v']).reshape(grid_z_size)))
                    elif ('extra' in _config['OUT']['overlay_type']):
                        if (fidx < len(PIV_data)):
                            extra_idx = FirstIntInStr(_config['OUT']['overlay_type'])
                            if (extra_idx < len(PIV_data[fidx]['extra'])):
                                if (fidx < len(PIV_data)):
                                    overlay_plot = np.transpose(np.asarray(PIV_data[fidx]['extra'][extra_idx]).reshape(grid_size))
                                if (PIVz_data is not None and len(global_drop_idx)>0):
                                    for dridx in range(len(figz_list)):
                                        if (global_drop_idx[dridx] in PIVz_idx):
                                            overlay_z_plots.append(np.transpose(np.asarray(PIVz_data[PIVz_idx.index(global_drop_idx[dridx])]['v']).reshape(grid_z_size)))
                    

                    if (OUT_folder is not None):
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            
                            if (chEdgeUp is not None):
                                if (ax is not None):
                                    ax.fill_between(chEdgeUp[:,0], 0, chEdgeUp[:,1], **_config['OUT']['ch_edge_shade'])
                                    ax.plot(chEdgeUp[:,0], chEdgeUp[:,1], '-', **_config['OUT']['ch_edge_line'])
                                if (figc_axm is not None):
                                    figc_axm.fill_between(chEdgeUp[:,0], 0, chEdgeUp[:,1], **_config['OUT']['ch_edge_shade'])
                                    figc_axm.plot(chEdgeUp[:,0], chEdgeUp[:,1], '-', **_config['OUT']['ch_edge_line'])
                            if (chEdgeDwn is not None):
                                if (ax is not None):
                                    ax.fill_between(chEdgeDwn[:,0], chEdgeDwn[:,1], img_arr_clip.shape[0]+1, **_config['OUT']['ch_edge_shade'])
                                    ax.plot(chEdgeDwn[:,0], chEdgeDwn[:,1], '-', **_config['OUT']['ch_edge_line'])
                                if (figc_axm is not None):
                                    figc_axm.fill_between(chEdgeDwn[:,0], chEdgeDwn[:,1], img_arr_clip.shape[0]+1, **_config['OUT']['ch_edge_shade'])
                                    figc_axm.plot(chEdgeDwn[:,0], chEdgeDwn[:,1], '-', **_config['OUT']['ch_edge_line'])
                            
                            if (overlay_plot is not None):
                                my_cmap = _config['OUT']['overlay_cmap']                    
                                alphas = Normalize(clip=True)(np.power(np.nan_to_num(overlay_plot), _config['OUT']['overlay_alpha_pwr']))   # Create an alpha channel based on weight values. Any value whose absolute value is > .0001 will have zero transparency
                                if (_config['OUT']['overlay_alpha_clip'] is not None):
                                    alphas = np.clip(alphas, _config['OUT']['overlay_alpha_clip'][0], _config['OUT']['overlay_alpha_clip'][1])  # alpha value clipped at the bottom at .4
                                colors = Normalize(vmin=np.nanmin(overlay_plot), vmax=np.nanmax(overlay_plot))(overlay_plot) # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                                colors = my_cmap(colors)
                                colors[..., -1] = alphas   # Now set the alpha channel to the one we created above
                                cur_imshow_kw = {'extent' : [PIV_avg['x'][0]*0.5, PIV_avg['x'][-1]+PIV_avg['x'][0]*0.5,\
                                                               PIV_avg['y'][-1]+PIV_avg['y'][0]*0.5, PIV_avg['y'][0]*0.5],}
                                if (_config['OUT']['overlay_vbounds'] is not None):
                                    cur_imshow_kw['vmin'] = _config['OUT']['overlay_vbounds'][0]
                                    cur_imshow_kw['vmax'] = _config['OUT']['overlay_vbounds'][1]
                                if (ax is not None):
                                    ax.imshow(colors, **cur_imshow_kw)
                                if (figc_axm is not None):
                                    figc_axm.imshow(colors, **cur_imshow_kw)
                            for dridx in range(min(len(axz_list), len(overlay_z_plots))):
                                my_cmap = _config['OUT']['overlay_cmap']                    
                                alphas = Normalize(clip=True)(np.power(np.nan_to_num(overlay_z_plots[dridx]), _config['OUT']['overlay_alpha_pwr']))   # Create an alpha channel based on weight values. Any value whose absolute value is > .0001 will have zero transparency
                                if (_config['OUT']['overlay_alpha_clip'] is not None):
                                    alphas = np.clip(alphas, _config['OUT']['overlay_alpha_clip'][0], _config['OUT']['overlay_alpha_clip'][1])  # alpha value clipped at the bottom at .4
                                colors = Normalize(vmin=np.nanmin(overlay_z_plots[dridx]), vmax=np.nanmax(overlay_z_plots[dridx]))(overlay_z_plots[dridx]) # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
                                colors = my_cmap(colors)
                                colors[..., -1] = alphas   # Now set the alpha channel to the one we created above
                                axz_list[dridx].imshow(colors, extent=[PIVz_avg['x'][0]*0.5, PIVz_avg['x'][-1]+PIVz_avg['x'][0]*0.5,\
                                                           PIVz_avg['y'][-1]+PIVz_avg['y'][0]*0.5, PIVz_avg['y'][0]*0.5])
                                if (figc_axz is not None and dridx == _config['OUT']['combined_zoom_idx']):
                                    figc_axz.imshow(colors, extent=[PIVz_avg['x'][0]*0.5, PIVz_avg['x'][-1]+PIVz_avg['x'][0]*0.5,\
                                                           PIVz_avg['y'][-1]+PIVz_avg['y'][0]*0.5, PIVz_avg['y'][0]*0.5])
                            
                            if (_config['OUT']['quiveropts'] is not None and PIV_data is not None):
                                if (fidx < len(PIV_data)):
                                    #PIV_data[fidx]['u'] = np.ma.masked_where(PIV_data[fidx]['u'], np.ma.mask_or(np.ma.make_mask(PIV_data[fidx]['u']==0), np.ma.getmask(PIV_data[fidx]['u'])))
                                    u_masked = np.ma.masked_where(PIV_data[fidx]['u']==0, PIV_data[fidx]['u'])
                                    if (ax is not None):
                                        Q = ax.quiver(PIV_data[fidx]['x'], PIV_data[fidx]['y'], u_masked, np.multiply(PIV_data[fidx]['v'], -1), **_config['OUT']['quiveropts'])
                                        if (_config['OUT']['quiverkey'] is not None):
                                            qk = ax.quiverkey(Q, _config['OUT']['quiverkey'][0], _config['OUT']['quiverkey'][1],\
                                                               _config['OUT']['quiverkey'][2], _config['OUT']['qk_label'], labelpos='E', coordinates='figure')
                                    if (figc_axm is not None):
                                        Qc = figc_axm.quiver(PIV_data[fidx]['x'], PIV_data[fidx]['y'], u_masked, np.multiply(PIV_data[fidx]['v'], -1), **_config['OUT']['quiveropts'])
                                        if (_config['OUT']['quiverkey'] is not None):
                                            qkc = figc_axm.quiverkey(Qc, _config['OUT']['quiverkey'][0], _config['OUT']['quiverkey'][1],\
                                                               _config['OUT']['quiverkey'][2], _config['OUT']['qk_label'], labelpos='E', coordinates='figure')
                            if (_config['OUT']['streamopts'] is not None and PIV_data is not None):
                                if ((ax is not None or figc_axm is not None) and fidx < len(PIV_data)):
                                    x_stream, y_stream = np.unique(np.asarray(PIV_data[fidx]['x']).flatten()), np.unique(np.asarray(PIV_data[fidx]['y']).flatten())
                                    if (_config['OUT']['stream_avg']):
                                        u_stream, v_stream = np.asarray(PIV_avg['u']).reshape(grid_size).T, np.asarray(PIV_avg['v']).reshape(grid_size).T
                                    else:
                                        u_stream, v_stream = PIV_data[fidx]['u'].reshape(grid_size).T, PIV_data[fidx]['v'].reshape(grid_size).T
                                    if (_config['OUT']['stream_seedstep'] is not None):
                                        y_seed = y_stream[::_config['OUT']['stream_seedstep']]
                                        seedpts = np.array([np.ones_like(y_seed)*x_stream[0], y_seed])
                                        _config['OUT']['streamopts']['start_points']=seedpts.T
                                    if (_config['OUT']['stream_color'] is not None):
                                        if (_config['OUT']['stream_color'] == 'norm'):
                                            _config['OUT']['streamopts']['color'] = np.hypot(u_stream, v_stream)
                                    u_stream_ma = np.ma.masked_where(u_stream==0, u_stream)
                                if (ax is not None and fidx < len(PIV_data)):
                                    ax.streamplot(x_stream, y_stream, u_stream_ma, v_stream, **_config['OUT']['streamopts'])
                                if (figc_axm is not None and fidx < len(PIV_data)):
                                    figc_axm.streamplot(x_stream, y_stream, u_stream_ma, v_stream, **_config['OUT']['streamopts'])
                            if (_config['OUT']['quiveropts'] is not None and PIVz_data is not None):
                                for dridx in range(min(len(axz_list), len(global_drop_idx))):
                                    PIVz_cur_index = -1
                                    #print('Processing droplet {0}...'.format(global_drop_idx[dridx]))
                                    if _config['PIVz']['idx_from_file']:
                                        if global_drop_idx[dridx] in PIVz_idx:
                                            PIVz_cur_index = PIVz_idx.index(global_drop_idx[dridx])
                                    else:
                                        if global_drop_idx[dridx] <= len(PIVz_idx):
                                            PIVz_cur_index = global_drop_idx[dridx]-1
                                            #print('...found in position {0}'.format(PIVz_cur_index))
                                        #else:
                                            #print('... NOT FOUND! {0} > {1}'.format(global_drop_idx[dridx], len(PIVz_idx)))
                                    if (PIVz_cur_index >= 0):
                                        Qz = axz_list[dridx].quiver(PIVz_data[PIVz_cur_index]['x'], PIVz_data[PIVz_cur_index]['y'], PIVz_data[PIVz_cur_index]['u'], np.multiply(PIVz_data[PIVz_cur_index]['v'], -1), **_config['OUT']['quiveropts'])
                                        if (_config['OUT']['quiverkey'] is not None):
                                            qkz = axz_list[dridx].quiverkey(Qz, _config['OUT']['quiverkey'][0], _config['OUT']['quiverkey'][1],\
                                                                _config['OUT']['quiverkey'][2], _config['OUT']['qk_label'], labelpos='E', coordinates='figure')
                                        if (figc_axz is not None and dridx == _config['OUT']['combined_zoom_idx']):
                                            Qzc = figc_axz.quiver(PIVz_data[PIVz_cur_index]['x'], PIVz_data[PIVz_cur_index]['y'], PIVz_data[PIVz_cur_index]['u'], np.multiply(PIVz_data[PIVz_cur_index]['v'], -1), **_config['OUT']['quiveropts'])
                                            if (_config['OUT']['quiverkey'] is not None):
                                                qkzc = figc_axz.quiverkey(Qzc, _config['OUT']['quiverkey'][0], _config['OUT']['quiverkey'][1],\
                                                                    _config['OUT']['quiverkey'][2], _config['OUT']['qk_label'], labelpos='E', coordinates='figure')
                                            
                            if (ax is not None):
                                ax.set_xlim([-0.5, img_arr_clip.shape[1]-0.5])
                                ax.set_ylim([img_arr_clip.shape[0]-0.5, -0.5])
                            if (figc_axm is not None):
                                figc_axm.set_xlim([-0.5, img_arr_clip.shape[1]-0.5])
                                figc_axm.set_ylim([img_arr_clip.shape[0]-0.5, -0.5])
                            if (_config['OUT']['hide_axes']):
                                if (ax is not None):
                                    ax.axis('off')
                                if (figc_axm is not None):
                                    figc_axm.axis('off')
                                if (figc_axz is not None):
                                    if (figc_axz_empty):
                                        figc_axz.axis('off')
                                    else:
                                        figc_axz.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                                        figc_axz.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                                        figc_axz.patch.set_edgecolor(_config['OUT']['rectangleopts']['edgecolor'])  
                                        figc_axz.patch.set_linewidth(4)  
                            if (fig is not None):
                                ax.set_position([0, 0, 1, 1])
                                fig.savefig(OUT_folder+_config['OUT']['figname_prefix']+str(my_idx).zfill(_config['OUT']['figname_idxlen'])+_config['OUT']['figname_ext'])
                            for dridx in range(len(figz_list)):
                                #figz_list[dridx].tight_layout()
                                axz_list[dridx].axis('off')
                                axz_list[dridx].set_position([0, 0, 1, 1])
                                figz_list[dridx].savefig(Zoom_folder+_config['OUT']['figname_prefix']+str(my_idx).zfill(_config['OUT']['figname_idxlen'])+\
                                                     '_'+str(dridx).zfill(2)+_config['OUT']['figname_ext'])
                            if (figc is not None):
                                figc.savefig(Combined_folder+_config['OUT']['figname_prefix']+str(my_idx).zfill(_config['OUT']['figname_idxlen'])+_config['OUT']['figname_ext'])
                    plt.close('all')
                
                
            if (find_logfile is not None):
                find_logfile.close()
        else:
        
            print('\nERROR: folder root {0} does not exist'.format(froot))
        
    plt.show()
