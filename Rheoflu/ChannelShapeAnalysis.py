import numpy as np

from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from PIL import Image

import Rheoflu.ChannelDesign as cd

def t_from_L(x, L, q, beta=1, zeta=1):
    return beta/(zeta*q)*np.concatenate([[0], np.cumsum(np.diff(x) * (L[:-1] + L[1:])/2)])
    
def st_from_Lt(xt, Lt):
    return -np.gradient(Lt, xt)/np.square(Lt)

def stress_from_shape(x, L, q, eta):
    return -q*eta*np.gradient(L, x)/np.square(L)

def q_from_L(t, x, L, beta=1, zeta=1, max_rtol=None):
    q = (beta/zeta)*L*np.gradient(x, t)
    if max_rtol is not None:
        rel_var = np.max(q) / np.min(q)-1
        if rel_val>max_rtol:
            print('WARNING: relative variation of {0:.3f}% exceeds threshold value {0:.3f}% in q_from_L'.format(rel_val*100, max_rtol*100))
    return np.mean(q)

def q_from_Lx(x, L, omega, beta=1, zeta=1):
    min_idx = argrelmin(L)[0]
    if len(min_idx)>1:
        t = t_from_L(x, L, q=1, beta=beta)
        return (t[min_idx[1]]-t[min_idx[0]])*omega/(2*np.pi)
    else:
        print('ERROR in q_from_Lx: provided channel shape needs to have at least one full period ({0})'.format(min_idx))
        return None

def dimensionless_wavelength(sigma_tilde):
    xt, _, _ = cd.solve_dimensionless(sigma_tilde, nperiods=1, npts=100)
    return xt[-1]-xt[0]

def dimensional_wavelength(omega, sigma, L0=1e-4, q=1e-4, eta=1e-3, beta=1, zeta=1):
    xt = dimensionless_wavelength(cd.calc_sigma_tilde(omega=omega, sigma=sigma, zeta=zeta, eta=eta))
    k = cd.calc_k(omega=omega, beta=beta, L0=L0, q=q, zeta=zeta)
    return xt/k

def dimensionless_amplitude(sigma_tilde):
    _, Lt, _ = cd.solve_dimensionless(sigma_tilde, nperiods=1, npts=1000)
    return np.max(Lt)-1

def dimensional_amplitude(omega, sigma, L0=1e-4, q=1e-4, eta=1e-3, zeta=1):
    amp_tilde = dimensionless_amplitude(cd.calc_sigma_tilde(omega=omega, sigma=sigma, zeta=zeta, eta=eta))
    return L0*amp_tilde
    
def channel_maxslope(omega, sigma, L0=1e-4, q=1e-4, eta=1e-3, beta=1):
    _, _, _, slope = cd.solve_dimensional(omega=omega, sigma=sigma, L0=L0, nperiods=0.5, npts=1000, q=q, eta=eta, beta=beta, return_slope=True)
    return np.max(np.abs(slope))

def sin_stress(t, amp, w, phi=0):
    return amp * np.sin(w * t + phi)

def sin_fit(t_data, s_data, nper=1):
    return curve_fit(sin_stress, t_data, s_data, p0=[max(s_data), 2*np.pi/(t_data[-1]-t_data[0])/nper, 0])
    
def analyze_sweep(x, L, t=None, q=None, q_from_omega=None, eta=1e-3, beta=1, zeta=1, include_extrema=True, silent=False):
    
    if t is None:
        if q is None and q_from_omega is not None:
            q = q_from_Lx(x, L, q_from_omega, beta=beta, zeta=zeta)
        if q is not None:
            t = t_from_L(x, L, q, beta=beta)
    if q is None and t is not None:
        q = q_from_L(t, x, L, beta=beta, zeta=zeta)
        
    if silent:
        min_idx = argrelmin(L)[0]
    else:
        fig, ax, ax2, min_idx = plot_channel(x, L, t=t, q=q, eta=eta, return_minima=True)
        print('Planar flow rate needed: {0:.1e} mm2/s'.format(q*1e6))
               
    stress_amps = []
    omega = []
    phi = []
    
    if len(min_idx)>0:
        all_s = stress_from_shape(x, L, q=q, eta=eta)
        for i in range(len(min_idx)+1):
            cur_s = None
            if i==0:
                if include_extrema:
                    cur_s = slice(0, min_idx[i])
            elif i < len(min_idx):
                cur_s = slice(min_idx[i-1], min_idx[i])
            else:
                if include_extrema:
                    cur_s = slice(min_idx[i-1], len(L))
            if cur_s is not None:
                curp, _ = sin_fit(t[cur_s], all_s[cur_s])
                if not silent:
                    ax2[0].plot(t[cur_s], sin_stress(t[cur_s], *curp), 'm--')
                stress_amps.append(np.abs(curp[0]))
                omega.append(np.abs(curp[1]))
                phi.append(np.abs(curp[2]))

        if not silent:
            if len(omega)>1:
                constr_idx = list(range(1, len(stress_amps)+1))
                fig3, ax3 = plt.subplots()
                ax3.plot(constr_idx, stress_amps, 'bs:', label=r'$\bar\sigma$')
                ax4 = ax3.twinx()
                ax4.plot(constr_idx, omega, 'go:', label=r'$\omega$')
                if (np.max(stress_amps)-np.min(stress_amps))>1e-2*np.mean(stress_amps):
                    ax3.set_yscale('log')
                else:
                    ax3.set_ylim([np.mean(stress_amps)-1e-2*np.mean(stress_amps), np.mean(stress_amps)+1e-2*np.mean(stress_amps)])
                if (np.max(omega)-np.min(omega))>1e-2*np.mean(omega):
                    ax4.set_yscale('log')
                else:
                    ax4.set_ylim([np.mean(omega)-1e-2*np.mean(omega), np.mean(omega)+1e-2*np.mean(omega)])
                ax3.set_xlabel('constriction #')
                ax3.set_ylabel(r'$\bar\sigma$ [Pa]')
                ax4.set_ylabel(r'$\omega$ [rad/s]')
                fig3.legend()
            elif len(omega)>0:
                print('Single constriction detected:')
                print('omega = {0:.1f} [rad/s]'.format(omega[0]))
                print('sigma = {0:.3f} [Pa]'.format(stress_amps[0]))
                print('phi   = {0:.3f} [rad]'.format(phi[0]))
        
    return stress_amps, omega, phi

# Channel plotting function

def plot_channel(x, L, t=None, q=None, omega=None, eta=1e-3, beta=1, zeta=1, return_minima=False):
    """
    if t is None, try to reconstruct it based on:
    - q, if given
    - otherwise omega (single float value), if there are at least two constrictions in L(x)
      (not compatible with freq sweep)
    """
    
    min_idx = argrelmin(L)
    
    if t is None:
        if q is None and omega is not None:
            q = q_from_Lx(x, L, omega=omega, beta=beta, zeta=zeta)
        if q is not None:
            t = t_from_L(x, L, q, beta=beta)
    else:
        if omega is not None:
            print('WARNING: disregarding omega parameter in plot_channel, as t is specified')

    if t is None:
        fig, ax = plt.subplots()
        cax = ax
    else:
        fig, ax = plt.subplots(ncols=2, figsize=(10,6))
        cax = ax[0]
        ax[1].plot(x*1e3, L/2*1e6, 'k-')
        ax[1].plot(x*1e3, -L/2*1e6, 'k-')
        ax[1].plot(x[min_idx]*1e3, L[min_idx]/2*1e6, 'kx')
        ax[1].set_xlabel(r'$x$ [mm]')
    cax.plot(t, L/2*1e6, 'k-')
    cax.plot(t, -L/2*1e6, 'k-')
    cax.plot(t[min_idx], L[min_idx]/2*1e6, 'kx')
    cax.set_xlabel(r'$t$ [s]')
    cax.set_ylabel(r'$L$ [µm]')
    
    if q is not None:
        ax2 = [cax.twinx() for cax in ax]
        stress = stress_from_shape(x, L, q=q, eta=eta)
        ax2[0].plot(t, stress, 'r:')
        ax2[1].plot(x*1e3, stress, 'r:')
        ax2[0].plot(t[min_idx], stress[min_idx], 'r+')
        ax2[1].plot(x[min_idx]*1e3, stress[min_idx], 'r+')
        ax2[1].set_ylabel(r'$\sigma$ [Pa]')
        ax2[0].axes.get_yaxis().set_ticklabels([])
        ax[1].axes.get_yaxis().set_ticklabels([])
    else:
        ax2 = None

    fig.tight_layout()
    
    if return_minima:
        return fig, ax, ax2, min_idx[0]
    else:
        return fig, ax, ax2
    
def plot_solution(omega, sigma, q, eta, L0, beta=1, zeta=1):
    x, L, t = cd.solve_dimensional(omega=omega, sigma=sigma, q=q, eta=eta, L0=L0)
    fig, ax, ax2 = plot_channel(x, L, t=t, q=q, eta=eta)
    ax[1].axvline(x=dimensional_wavelength(omega=omega, sigma=sigma, q=q, eta=eta, L0=L0)*1e3, c='b', ls='--')
    ax[1].axhline(y=0.5*(L0+dimensional_amplitude(omega=omega, sigma=sigma, q=q, eta=eta, L0=L0))*1e6, c='b', ls='--')
    
    
def AnalyzeChannelShape(channel_img, topedge, bottomedge, crop=None, px_size=1, q=None, design_omega=300, eta=1.7e-2, spl_smoothf=10, beta=1, zeta=1, save_fig=None):
    if isinstance(channel_img, str):
        img_arr = np.array(Image.open(channel_img), dtype=float)
    else:
        img_arr = channel_img
    if crop is None:
        crop = [0, 0, 0, 0]
    if (crop[2] <= 0):
        crop[2] = img_arr.shape[1]+crop[2]
    if (crop[3] <= 0):
        crop[3] = img_arr.shape[0]+crop[3]
    if isinstance(topedge, str):
        topedge = np.loadtxt(topedge)
    if isinstance(bottomedge, str):
        bottomedge = np.loadtxt(bottomedge)
    topedge *= px_size
    bottomedge *= px_size

    spl_top = UnivariateSpline(topedge[:,0], topedge[:,1], s=spl_smoothf)
    spl_bottom = UnivariateSpline(bottomedge[:,0], bottomedge[:,1], s=spl_smoothf)
    spl_xarr = np.arange(int(img_arr.shape[1]*px_size))
    
    x, L = 1e-6*spl_xarr, 1e-6*np.abs(spl_top(spl_xarr) - spl_bottom(spl_xarr))
    if q is None:
        q = q_from_Lx(x, L, design_omega, beta=beta, zeta=zeta)
    else:
        design_omega = None
    t = t_from_L(x, L, q=q)
    constriction_params = analyze_sweep(x, L, q=q, q_from_omega=design_omega, eta=eta, include_extrema=False, silent=True)
    stramp, omega, phi = constriction_params[0][0], constriction_params[1][0], constriction_params[2][0]
    if design_omega is None:
        design_omega = omega
    
    L0 = 1e-6*np.min(np.abs(spl_top(spl_xarr) - spl_bottom(spl_xarr)))
    chaxis = 0.5*(spl_top(spl_xarr) + spl_bottom(spl_xarr))
    
    out_params = {'stress_amp': stramp, 
                  'omega'     : omega, 
                  'phi'       : phi, 
                  'L0'        : L0, 
                  'q'         : q}
    
    min_idx = argrelmin(L)[0]
    minpos = [float(x[i]) for i in min_idx]
    minpos_px = [int(x*1e6/px_size) for x in minpos]
    
    fig, ax = plt.subplots(figsize=(20,5))
    ax.imshow(img_arr, cmap='Greys_r', extent=np.multiply(px_size, [0, img_arr.shape[1], 0, img_arr.shape[0]]), origin='lower', aspect='auto')
    ax.plot(topedge[:,0], topedge[:,1], 'c+')
    ax.plot(bottomedge[:,0], bottomedge[:,1], 'bx')

    ax.plot(spl_xarr, spl_top(spl_xarr), 'c:', lw=2)
    ax.plot(spl_xarr, spl_bottom(spl_xarr), 'b:', lw=2)
    ax.plot(spl_xarr, chaxis, 'w:', lw=2)
    ax.fill_between(spl_xarr, spl_top(spl_xarr), img_arr.shape[0]*px_size, color='white', alpha=0.6)
    ax.fill_between(spl_xarr, spl_bottom(spl_xarr), 0, color='white', alpha=0.6)
    for xmin in minpos:
        ax.axvline(x=xmin*1e6, c='k', ls=':')
    
    ax.set_xlim([px_size*crop[0],px_size*crop[2]])
    ax.set_ylim([px_size*crop[1],px_size*crop[3]])

    ax.set_ylabel(r'$y$ [µm]')
    ax.set_xlabel(r'$x$ [µm]')
    
    ax2 = ax.twinx()
    ax2.plot(x*1e6, stress_from_shape(x, L, q=q, eta=eta), 'm-', lw=2, label=r'$\sigma=-q\eta L^\prime/L^2$')
    ax2.plot(x[min_idx[0]:min_idx[1]]*1e6, sin_stress(t[min_idx[0]:min_idx[1]], stramp, omega, phi), 'y--', lw=2, label=r'$\bar\sigma_f \sin(\omega_f t+\phi_f)$')
    ax2.plot(x*1e6, sin_stress(t-t[min_idx[0]], stramp, design_omega, phi=np.pi), 'r:', lw=4, label=r'$\bar\sigma \sin(\omega (t-t_0))$')
    ax2.legend()
    ax2.set_ylabel(r'$\sigma$ [Pa]')
    ax2.set_ylim([-1.2*stramp,1.2*stramp])
    
    if save_fig is not None:
        fig.savefig(save_fig)
    
    print('Constriction parameters:')
    print('omega = {0:.1f} [rad/s]'.format(omega))
    print('sigma = {0:.3f} [Pa]'.format(stramp))
    print('phi   = {0:.3f} [rad]'.format(phi))
    print('q     = {0:.3f} [mm2/s]'.format(q*1e6))
    print('L0    = {0:.1f} [um]'.format(L0*1e6))
    if len(min_idx) > 1:
        wavelength = minpos[1] - minpos[0]
        print('wavelength = {0:.1f} [um]'.format(wavelength*1e6))
        out_params['wavelength_um'] = wavelength
    else:
        print('wavelength UNDETERMINED')
        out_params['wavelength_um'] = None
    
    return out_params, [x, L, t], [spl_top, spl_bottom], int(np.mean(chaxis)/px_size), minpos_px