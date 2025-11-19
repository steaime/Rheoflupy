import os
import logging
import json
import datetime
import numpy as np
import tifffile
import cv2

def load_params(param_fpath, kwargs):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    defpar_path = os.path.join(script_dir, 'default_params.txt')
    with open(defpar_path, 'r') as f:
        def_params = json.load(f)
        print('Default analysis parameters loaded from configuration file: ' + defpar_path)
    if os.path.isfile(param_fpath):
        with open(param_fpath, 'r') as f:
            params = json.load(f)
            print('Analysis parameters loaded from configuration file: ' + param_fpath)
    for k, val in kwargs.items():
        if k in params:
            print(' - param[{0}] updated from {1} to {2} using function kwargs'.format(k, params[k], kwargs[k]))
        else:
            print(' - param[{0}]={1} added using function kwargs'.format(k, kwargs[k]))
        params[k] = kwargs[k]
    def_keys = [k for k in def_params if k not in params]
    if len(def_keys)>0:
        print('{0} parameters absent in parameter file - default parameters used instead'.format(len(def_keys)))
        for k in def_keys:
            params[k] = def_params[k]
            print(' - {0} : {1}'.format(k, params[k]))
    return params

def setup_logger(logfpath):
    fout = open(logfpath, 'a')
    fout.write('\nLogger started on: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return fout

def close_logger(flog):
    flog.write('\nLogger ended on: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    flog.close()
    
def printlog(smsg, flog=None, flush=True, prefix='\n'):
    print(smsg)
    if flog is not None:
        flog.write(prefix + smsg)
        if flush:
            flog.flush()

def get_stack_headlen(fpath):
    with tifffile.TiffFile(fpath) as tif:
        off_bytes = tif.pages[0].dataoffsets
    return off_bytes

def get_stack_shape(fpath):
    with tifffile.TiffFile(fpath) as tif:
        n_pages = len(tif.pages)
        frame = tif.pages[0].asarray()  # Only reads this page into memory
    return (n_pages, *frame.shape)
    
def compute_background(fpath, avg_range=None):
    with tifffile.TiffFile(fpath) as tif:
        n_pages = len(tif.pages)
        res = np.zeros_like(tif.pages[0].asarray(), dtype=float)
        count = 0
        if avg_range is None:
            avg_range = [n_pages]
        for i in range(*avg_range):
            if i < n_pages:
                res += tif.pages[i].asarray()
                count += 1
    return res / count
        
def get_single_frame(fpath, frame_n, cropROI=None, bkg=None, bkgcorr_offset=0, blur_sigma=0, blur_kernel=(0,0), dtype=np.uint8):
    return get_stack(fpath, frame_range=[frame_n, frame_n+1], cropROI=cropROI, bkg=bkg, bkgcorr_offset=bkgcorr_offset, 
                     blur_sigma=blur_sigma, blur_kernel=blur_kernel, dtype=dtype)[0]

def get_stack(fpath, frame_range, cropROI=None, bkg=None, bkgcorr_offset=0, blur_sigma=0, blur_kernel=(0,0), dtype=np.uint8):
    res = None
    with tifffile.TiffFile(fpath) as tif:
        if frame_range is None:
            frame_range = [n_pages]
        sel_frames = list(range(*frame_range))
        img_shape = tif.pages[0].asarray().shape
        if cropROI is None:
            cropROI = [0,0,img_shape[1],img_shape[0]]
        if (cropROI[2] <= 0):
            cropROI[2] = res.shape[1]+cropROI[2]
        if (cropROI[3] <= 0):
            cropROI[3] = res.shape[0]+cropROI[3]
        res = np.empty((len(sel_frames), cropROI[3]-cropROI[1], cropROI[2]-cropROI[0]), dtype=dtype)
        for i in range(len(sel_frames)):
            if sel_frames[i] < len(tif.pages):
                cur_frame = tif.pages[sel_frames[i]].asarray()
                if bkg is not None:
                    cur_frame = cur_frame - bkg + bkgcorr_offset
                if blur_sigma > 0:
                    cv2.GaussianBlur(cur_frame, blur_kernel, blur_sigma)
                res[i] = cur_frame[cropROI[1]:cropROI[3],cropROI[0]:cropROI[2]]
    return res

def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        os.makedirs(folderPath)
        print("Created folder: {0}".format(folderPath))
        return False