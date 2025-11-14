import os
import numpy as np
import tifffile
import cv2

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