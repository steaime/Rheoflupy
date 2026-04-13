import functools
import numpy as np
from scipy.integrate import solve_ivp

# Channel shape design functions (single stress and frequency)

def dLdt(t, L, sigma, omega, q, eta): 
    return L * (sigma / eta) * np.sin(omega * t)

def dL_dx(t, L, sigma, omega, q, eta): 
    return L**2 * (sigma / (q*eta)) * np.sin(omega * t)

def dLt_dtt(tt, Lt, st=1.):
    return Lt*st*np.sin(tt)

def Lt_tt(tt, st=1.):
    sol = solve_ivp(functools.partial(dLt_dtt, st=st), (tt[0], tt[-1]), [1.], t_eval=tt, rtol=1e-8, atol=1e-8)
    return sol.y[0]

def xt_tt(tt, Lt):
    return np.concatenate([[0], np.cumsum(np.diff(tt) * 2/(Lt[:-1] + Lt[1:]))])

def calc_sigma_tilde(omega, sigma, eta=1e-3, zeta=1):
    return sigma*zeta/(eta*omega)

def calc_k(omega, beta, L0, q, zeta=1):
    return omega*beta*L0/(q*zeta)

def q_from_k(omega, beta, L0, k, zeta=1):
    return omega*beta*L0/(k*zeta)

def solve_dimensionless(sigma_tilde, nperiods=2, npts=1000):
    tt = np.linspace(0, 2*np.pi*nperiods, npts)
    Lt = Lt_tt(tt, st=sigma_tilde)
    xt = xt_tt(tt, Lt)
    return xt, Lt, tt

def solve_dimensional(omega, sigma, L0=1e-4, nperiods=2, npts=1000, q=1e-4, eta=1e-3, beta=1, zeta=1, return_slope=False):
    xt, Lt, tt = solve_dimensionless(sigma_tilde=calc_sigma_tilde(omega=omega, sigma=sigma, zeta=zeta, eta=eta), 
                                     nperiods=nperiods, npts=npts)
    t, x, L = tt/omega, xt*q*zeta/(omega*beta*L0), L0*Lt
    if return_slope:
        dLdx = np.gradient(L, x)
        return x, L, t, dLdx
    else:
        return x, L, t
    
def creep_length_dimensionless(sigma_tilde, tmax_tilde=1):
    return (1./sigma_tilde) * (1-np.exp(-tmax_tilde))

def creep_dimensionless(sigma_tilde, tmax_tilde=1, npts=1000):
    xt_max = creep_length_dimensionless(sigma_tilde, tmax_tilde)
    xt = np.linspace(0, xt_max, npts)
    Lt = 1/(1-sigma_tilde*xt)
    return xt, Lt
    
def creep_length(sigma, tmax, L0=1e-4, q=1e-4, eta=1e-3, beta=1, zeta=1):
    return L0 * creep_length_dimensionless(sigma_tilde=beta*sigma*L0**2/(q*eta), tmax_tilde=zeta*sigma*tmax/eta)     
            # = (q*eta / (beta*sigma*L0)) * (1 - np.exp(-zeta*sigma*tmax/eta))
    
def creep_dimensional(sigma, tmax, L0=1e-4, npts=1000, q=1e-4, eta=1e-3, beta=1, zeta=1):
    #xmax = creep_length(sigma, tmax, L0=L0, q=q, eta=eta, beta=beta, zeta=zeta)
    #x = np.linspace(0, xmax, npts)
    #L = L0 / (1 - x * beta*sigma*L0/(q*eta))
    xt, Lt = creep_dimensionless(sigma_tilde=beta*sigma*L0**2/(q*eta), tmax_tilde=zeta*sigma*tmax/eta, npts=npts)
    return xt*L0, Lt*L0

def solve_generalized(t, sigma, L0=1e-4, npts=1000, q=1e-4, eta=1e-3, beta=1, zeta=1):
    return
    
# Channel shape design functions (multiple stress/freq)

def concatenate_dimensionless(sigma_tilde_list, rel_k_list=None, nperiods=2, pts_per_sol=1000):
    for i in range(len(sigma_tilde_list)):
        xt, Lt, tt = solve_dimensionless(sigma_tilde_list[i], nperiods=nperiods, npts=pts_per_sol)
        if rel_k_list is not None:
            if len(rel_k_list) > i:
                tt /= rel_k_list[i]
                xt /= rel_k_list[i]
        if i==0:
            all_tt, all_xt, all_Lt = tt, xt, Lt
        else:
            all_tt = np.concatenate([all_tt, tt[1:]+all_tt[-1]])
            all_xt = np.concatenate([all_xt, xt[1:]+all_xt[-1]])
            all_Lt = np.concatenate([all_Lt, Lt[1:]])            
    return all_xt, all_Lt, all_tt

def gen_param_list(omega, sigma, L0=1e-4, nperiods=2, pts_per_sol=1000, q=1e-4, eta=1e-3, beta=1, zeta=1):
    if not hasattr(omega, '__len__'):
        omega = [omega] * len(sigma)
    if not hasattr(sigma, '__len__'):
        sigma = [sigma] * len(omega)
    noscill = min(len(omega), len(sigma))
    params = [{'omega'   : omega[i], 
               'sigma'   : sigma[i], 
               'L0'      : L0, 
               'nperiods': nperiods, 
               'pts'     : pts_per_sol, 
               'q'       : q, 
               'eta'     : eta,
               'beta'    : beta,
               'zeta'    : zeta,
               'sigma_tilde' : calc_sigma_tilde(omega=omega[i], sigma=sigma[i], eta=eta, zeta=zeta),
               'k'       : calc_k(omega=omega[i], beta=beta, L0=L0, q=q, zeta=zeta)
               } 
              for i in range(noscill)]
    return params

def channel_shape(omega, sigma, L0=1e-4, nperiods=2, pts_per_sol=1000, q=1e-4, eta=1e-3, beta=1, zeta=1, return_params=False):
    params = gen_param_list(omega=omega, sigma=sigma, L0=L0, nperiods=nperiods, 
                            pts_per_sol=pts_per_sol, q=q, eta=eta, beta=beta, zeta=zeta)
    for i in range(len(params)):
        p = params[i]
        x, L, t = solve_dimensional(omega=p['omega'], sigma=p['sigma'], L0=p['L0'], nperiods=p['nperiods'], 
                                    npts=p['pts'], q=p['q'], eta=p['eta'], beta=p['beta'], zeta=p['zeta'])
        if i==0:
            all_t, all_x, all_L = t, x, L
        else:
            all_t = np.concatenate([all_t, t[1:]+all_t[-1]])
            all_x = np.concatenate([all_x, x[1:]+all_x[-1]])
            all_L = np.concatenate([all_L, L[1:]])
    if return_params:
        return all_x, all_L, all_t, params
    else:
        return all_x, all_L, all_t

# Channel shape design functions (set total length)

def sweep_setlength(sigma_tilde_list, rel_k_list=None, channel_length=1, L0=1, omega_scale=1, nperiods=2, pts_per_sol=1000, return_k=False):
    xt, Lt, tt = concatenate_dimensionless(sigma_tilde_list, rel_k_list=rel_k_list, nperiods=nperiods, pts_per_sol=pts_per_sol)
    k_scale = (xt[-1]-xt[0])/channel_length
    if return_k:
        return xt/k_scale, Lt*L0, tt/omega_scale, k_scale
    else:
        return xt/k_scale, Lt*L0, tt/omega_scale
    
def stress_sweep(omega, sigma, channel_length, L0=1e-4, nperiods=2, pts_per_sol=1000, eta=1e-3, beta=1, zeta=1, return_q=False):
    sigma_tilde_list = [calc_sigma_tilde(omega=omega, sigma=s, eta=eta, zeta=zeta) for s in sigma]
    x, L, t, k = sweep_setlength(sigma_tilde_list, channel_length=channel_length, L0=L0, omega_scale=omega, nperiods=nperiods, pts_per_sol=pts_per_sol, return_k=True)
    if return_q:
        calc_q = q_from_k(omega=omega, beta=beta, L0=L0, k=k, zeta=zeta)
        return x, L, t, calc_q
    else:
        return x, L, t
    
def sweep_setlength_absk(sigma_tilde_list, rel_k_list, channel_length=1, L0=1, omega_scale=1, eta=1e-3, beta=1, zeta=1, nperiods=2, pts_per_sol=1000, max_iter=10, verbose=0, return_k=False):
    k_in = rel_k_list
    w_scale = omega_scale
    kscale_list = []
    for j in range(max_iter):
        test_x, test_L, test_t, kscale_out = sweep_setlength(sigma_tilde_list, rel_k_list=k_in, channel_length=channel_length, L0=L0, 
                                                             omega_scale=w_scale, nperiods=nperiods, pts_per_sol=pts_per_sol, return_k=True)
        kscale_list.append(kscale_out)
        k_logrelerr = np.log10(np.abs(kscale_out-1))
        if k_logrelerr>-2:
            for i in range(len(k_in)):
                k_in[i] *= kscale_out
            w_scale /= kscale_out
        else:
            if verbose > 0:
                print('{0} iterations: k scale is off by {1:.1e}%'.format(len(kscale_list), 100*10**k_logrelerr))
            break
    if verbose > 1:
        plt.plot(kscale_list)

    if return_k:
        return test_x, test_L, test_t, k_in, 
    else:
        return test_x, test_L, test_t

def channel_set_length(omega, sigma, channel_length, L0=1e-4, nperiods=2, pts_per_sol=1000, eta=1e-3, beta=1, zeta=1, max_iter=100, return_params=False):
    pars = gen_param_list(omega=omega, sigma=sigma, L0=L0, nperiods=nperiods, pts_per_sol=pts_per_sol, eta=eta, beta=beta, zeta=zeta)
    sigma_tilde_list = [p['sigma_tilde'] for p in pars]
    rel_k_list = [p['k']/pars[0]['k'] for p in pars]
    x, L, t, k_list = sweep_setlength_absk(sigma_tilde_list=sigma_tilde_list, rel_k_list=rel_k_list, 
                                           channel_length=channel_length, L0=L0, omega_scale=pars[0]['omega'],
                                           nperiods=nperiods, pts_per_sol=pts_per_sol, return_k=True)
    for i in range(len(pars)):
        pars[i]['k'] = k_list[i]
        pars[i]['q'] = q_from_k(omega=pars[i]['omega'], beta=beta, L0=L0, k=pars[i]['k'], zeta=zeta)
    q_list = [p['q'] for p in pars]
    if np.log10(np.abs(np.max(q_list)/np.min(q_list)-1) > -3):
        print('WARNING: q values computed using k and omega for each constriction vary by more than 0.1%')
    if return_params:
        return x, L, t, pars
    else:
        return x, L, t
