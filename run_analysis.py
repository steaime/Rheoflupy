import sys
import os
import Rheoflu

params_fpath = os.path.join('Rheoflu', 'default_params.txt')

if __name__ == "__main__":
    
    if len(sys.argv)>1:
        if os.path.isfile(sys.argv[1]):
            params_fpath = sys.argv[1]

    if False:
        for d in [25, 35]:
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_500h_acmc_20240911_110922_PM_20240911_110950_PM', fps=4204, filter_d=d)
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_800h_acmc_20240911_110229_PM_20240911_110255_PM', fps=4204, filter_d=d)
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_1000h_acmc_20240911_105640_PM_20240911_105654_PM', fps=4204, filter_d=d)
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_2000h_acmc_20240911_103709_PM_20240911_103857_PM', fps=4204, filter_d=d)
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_2000h_acmc_20240911_104620_PM_20240911_104648_PM', fps=4204, filter_d=d)
            Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_7000ulh_20240815_52604_PM_20240815_052657_PM_rotated', fps=4954, filter_d=d)
        
    Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_500ulh_20240815_51852_PM_20240815_051939_PM_rotated', fps=4954, 
                             filter_d=21, track_minmass=1000, filter_range=[5, 50], filter_ss_maxtomean=10, allowed_badpoints=10)
    Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_2000ulh_20240815_50923_PM_20240815_051110_PM_rotated', fps=4954, 
                            filter_d=21, track_minmass=1000, filter_range=[5, 50], filter_ss_maxtomean=10, allowed_badpoints=10)
    Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_10000ulh_20240815_54215_PM_20240815_054339_PM_rotated', fps=4954, 
                            filter_d=21, track_minmass=1000, filter_range=[5, 50], filter_ss_maxtomean=10, allowed_badpoints=10)
    Rheoflu.rheoflu_analysis(params_fpath, img_name='tween80_10000ulh_20240815_54915_PM_20240815_055020_PM_rotated', fps=4954, 
                            filter_d=21, track_minmass=1000, filter_range=[5, 50], filter_ss_maxtomean=10, allowed_badpoints=10)