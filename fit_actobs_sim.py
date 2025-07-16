"""Bayesian fitting module.

This module implements Bayesian parameter estimation. The model describes how 
observers update their beliefs about the state of the world based on 
sensory evidence and make decisions about when to sample new information.

The main function `fit_actobs_sim` fits model parameters to behavioral data
using Bayesian Adaptive Direct Search (BADS) optimization.

Key model parameters:
- h: Hazard rate (probability of state change)
- sinf: Inference noise (uncertainty in belief updates)
- ssel: Selection noise (uncertainty in decision making)
- scnf: Confidence noise (uncertainty in confidence reports)
- tcnf: Confidence threshold (bias in confidence reports)
- gcnf: Confidence gain (scaling of confidence during switches)
"""

import numpy as np
from scipy.stats import beta, gamma, norm
from pybads.bads import BADS
from getc import getc
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings


def estimate_ll_sd(phat, cfg, nres=1000):
    """Bootstrap standard deviation of the log-likelihood"""
    nsmp = cfg['nsmp']
    lmin = 0.5 / nsmp
    
    # Simulate responses using the same approach as in the main function
    def sim_r_local(pval):
        h = pval['h']
        sinf = pval['sinf']
        ssel = pval['ssel']
        scnf = pval.get('scnf', np.nan)
        tcnf = pval.get('tcnf', np.nan)
        gcnf = pval.get('gcnf', np.nan)
        
        nseq = len(cfg['seqind'])
        xt = np.zeros((nseq, nsmp))
        rbef = np.full((nseq, nsmp), np.nan)
        raft = np.full((nseq, nsmp), np.nan)
        cbef = np.full((nseq, nsmp), np.nan)
        caft = np.full((nseq, nsmp), np.nan)
        
        for iseq in range(nseq):
            if cfg['seqind'][iseq] > 1:
                rbef[iseq, :] = raft[iseq - 1, :]
                if cfg['fitcnf']:
                    cbef[iseq, :] = caft[iseq - 1, :]
            else:
                rbef[iseq, :] = cfg['rbef'][iseq]
                if cfg['fitcnf']:
                    cbef[iseq, :] = cfg['cbef'][iseq]
            
            if cfg['seqind'][iseq] > 1:
                xt[iseq, :] = upfun(xt[iseq - 1, :], h)
            
            xt[iseq, :] = np.random.normal(
                xt[iseq, :] + cfg['seqllr'][iseq],
                np.sqrt(cfg['seqlen'][iseq]) * sinf
            )
            
            xr = np.random.normal(xt[iseq, :], ssel)
            raft[iseq, :] = 1 + (xr < 0).astype(int)
            
            if cfg['fitcnf']:
                xc = xt[iseq, :] * (3 - 2 * raft[iseq, :])
                iswi = raft[iseq, :] != rbef[iseq, :]
                xc[iswi] = xc[iswi] * gcnf
                xc = np.random.normal(xc, scnf)
                caft[iseq, :] = 1 + (xc > tcnf).astype(int)
        
        return rbef, raft, cbef, caft
    
    rbef_sim, raft_sim, cbef_sim, caft_sim = sim_r_local(phat)
    ll_res = np.zeros(nres)
    for i in range(nres):
        jres = np.random.choice(nsmp, nsmp, replace=True)
        r_hat = np.mean(raft_sim[:,jres] == 1, axis=1)
        r_hat = (1 - 2 * lmin) * r_hat + lmin
        r_hat[cfg['raft'] == 2] = 1 - r_hat[cfg['raft'] == 2]
        ll = np.sum(np.log(r_hat))
        if cfg['fitcnf']:
            c_hat = np.mean(caft_sim[:,jres] == 1, axis=1)
            c_hat = (1 - 2 * lmin) * c_hat + lmin
            c_hat[cfg['caft'] == 2] = 1 - c_hat[cfg['caft'] == 2]
            ll += np.sum(np.log(c_hat))
        ll_res[i] = ll
    return max(np.std(ll_res), 1e-6)


def fit_actobs_sim(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fits computational model to behavioral data.
    
    This is the main function that fits a computational model to behavioral data. 
    The model describes how observers update
    their beliefs about environmental states and make decisions about when
    to sample new information.
    
    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary containing:
        
        Required experiment data:
        - 'seqind': Array of sequence indices in current block
        - 'seqpos': Array of sequence positions within current episode
        - 'seqdir': Array of sequence directions (1 or 2)
        - 'seqllr': Array of sequence log-likelihood ratios
        - 'seqlen': Array of sequence lengths
        
        Required response data:
        - 'rbef': Array of responses before each sequence
        - 'raft': Array of responses after each sequence
        
        Optional confidence data:
        - 'cbef': Array of confidence before each trial
        - 'caft': Array of confidence after each trial
        
        Optional fitting parameters:
        - 'fitrev': Fit reversal curves (default: False)
        - 'fitrep': Fit repetition curves (default: False)
        - 'fitcnf': Fit confidence data (auto-detected if not specified)
        - 'resamp': Enable resampling (default: False)
        - 'nsmp': Number of simulation samples (default: 1000)
        - 'nres': Number of bootstrap samples (default: 1000)
        - 'nval': Number of validation samples (default: 100)
        - 'nrun': Number of optimization runs (default: 10)
        - 'verbose': Verbosity level (0-2, default: 0)
        
        Optional fixed parameters (if not provided, will be fitted):
        - 'h': Hazard rate (0 < h < 1)
        - 'sinf': Inference noise (sinf > 0)
        - 'ssel': Selection noise (ssel > 0)
        - 'scnf': Confidence noise (scnf > 0)
        - 'tcnf': Confidence threshold
        - 'gcnf': Confidence gain (gcnf > 0)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing fitted results:
        
        Fitted parameters:
        - 'h': Fitted hazard rate
        - 'sinf': Fitted inference noise
        - 'ssel': Fitted selection noise
        - 'scnf': Fitted confidence noise (if fitcnf=True)
        - 'tcnf': Fitted confidence threshold (if fitcnf=True)
        - 'gcnf': Fitted confidence gain (if fitcnf=True)
        
        Optimization results:
        - 'xhat': Fitted parameter values (free parameters only)
        - 'xnam': Names of free parameters
        - 'll': Log-likelihood at fitted parameters
        - 'll_sd': Standard deviation of log-likelihood
        
        Model comparison metrics:
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        
        Fitting details:
        - 'nsmp': Number of simulation samples used
        - 'nval': Number of validation samples used
        - 'nrun': Number of optimization runs used
        - 'ntrl': Number of trials in dataset
        - 'nfit': Number of fitted parameters
        
        Data and curves:
        - 'cfg': Original configuration dictionary
        - 'csub': Subject behavioral curves
        - 'cfit': Fitted behavioral curves
        - 'psmp': Fraction of propagated samples (if resamp=True)
        - 'output': Raw optimization output
        
    Raises
    ------
    ValueError
        If required experiment data fields are missing
        If required response data fields are missing
        If confidence data is missing when fitcnf=True
        
    Notes
    -----
    The function uses Bayesian Adaptive Direct Search (BADS) for optimization.
    Multiple optimization runs are performed to avoid local minima.
    
    
    Examples
    --------
    Basic usage with minimal configuration:
    
    >>> cfg = {
    ...     'seqind': np.array([1, 2, 3, 4, 5]),
    ...     'seqpos': np.array([1, 2, 3, 4, 5]),
    ...     'seqdir': np.array([1, 1, 2, 2, 1]),
    ...     'seqllr': np.array([0.5, -0.3, 0.8, -0.2, 0.6]),
    ...     'seqlen': np.array([1, 1, 1, 1, 1]),
    ...     'rbef': np.array([1, 1, 1, 2, 2]),
    ...     'raft': np.array([1, 1, 2, 2, 1])
    ... }
    >>> result = fit_actobs_sim(cfg)
    >>> print(f"Fitted hazard rate: {result['h']:.3f}")
    
    Fitting with confidence data:
    
    >>> cfg['cbef'] = np.array([1, 1, 2, 2, 1])
    >>> cfg['caft'] = np.array([1, 2, 2, 1, 1])
    >>> cfg['fitcnf'] = True
    >>> result = fit_actobs_sim(cfg)
    >>> print(f"Confidence noise: {result['scnf']:.3f}")
    
    Fitting with fixed parameters:
    
    >>> cfg['h'] = 0.1  # Fix hazard rate
    >>> result = fit_actobs_sim(cfg)
    >>> print(f"Fixed hazard rate: {result['h']:.3f}")
    """
    # Check required experiment data fields
    required_exp_fields = ['seqind', 'seqpos', 'seqdir', 'seqllr', 'seqlen']
    if not all(key in cfg for key in required_exp_fields):
        raise ValueError('Missing experiment data!')
    
    # Check required response data fields
    required_resp_fields = ['rbef', 'raft']
    if not all(key in cfg for key in required_resp_fields):
        raise ValueError('Missing response data!')
    
    # Handle confidence data
    if not all(key in cfg for key in ['cbef', 'caft']):
        if cfg.get('fitcnf', False):
            raise ValueError('Missing confidence data!')
        cfg['cbef'] = np.full_like(cfg['rbef'], np.nan)
        cfg['caft'] = np.full_like(cfg['raft'], np.nan)
    
    # Set default configuration values
    cfg.setdefault('fitrev', False)
    cfg.setdefault('fitrep', False)
    cfg.setdefault('fitcnf', 
        all(key in cfg for key in ['cbef', 'caft']) and 
        not np.isnan(cfg['cbef']).any() and not np.isnan(cfg['caft']).any())
    cfg.setdefault('resamp', False)
    cfg.setdefault('nsmp', 1000)
    cfg.setdefault('nres', 1000)
    cfg.setdefault('nval', 100)
    cfg.setdefault('nrun', 10)
    cfg.setdefault('verbose', 0)
    
    # Convert to arrays and flatten
    for key in ['seqind', 'seqpos', 'seqdir', 'seqllr', 'seqlen']:
        cfg[key] = np.array(cfg[key]).flatten()
    
    nseq = len(cfg['seqind'])
    lmin = 0.5 / cfg['nsmp']
    
    # Do not fit confidence parameters if fitcnf is False
    if not cfg['fitcnf']:
        cfg['scnf'] = np.nan
        cfg['tcnf'] = np.nan
        cfg['gcnf'] = np.nan
    
    # Get subject reversal and repetition curves
    csub = getc(cfg['seqind'], cfg['seqpos'], cfg['seqdir'], cfg['seqllr'],
                cfg['rbef'], cfg['raft'], cfg['cbef'], cfg['caft'])

    priors = {
        'h': (1e-6, 1 - 1e-6, 0.125, beta.ppf(0.1587, 1, 7), beta.ppf(0.8413, 1, 7)),
        'sinf': (0, 10, 0.5, gamma.ppf(0.1587, 1, scale=0.5), gamma.ppf(0.8413, 1, scale=0.5)),
        'ssel': (0, 10, 1, gamma.ppf(0.1587, 1, scale=1), gamma.ppf(0.8413, 1, scale=1)),
        'scnf': (0, 10, 1, gamma.ppf(0.1587, 1, scale=1), gamma.ppf(0.8413, 1, scale=1)),
        'tcnf': (-10, 10, 0, norm.ppf(0.1587), norm.ppf(0.8413)),
        'gcnf': (0, 10, 1, gamma.ppf(0.1587, 2, scale=1), gamma.ppf(0.8413, 2, scale=1)),
    }

    fixed, free_names = {}, []
    bounds = {'x0': [], 'lb': [], 'ub': [], 'plb': [], 'pub': []}
    for name, (pmin, pmax, pini, pplb, ppub) in priors.items():
        val = cfg.get(name)
        if val is not None and not np.isnan(val):
            fixed[name] = val
        else:
            free_names.append(name)
            bounds['x0'].append(pini)
            bounds['lb'].append(pmin)
            bounds['ub'].append(pmax)
            bounds['plb'].append(pplb)
            bounds['pub'].append(ppub)

    def getpval(x):
        out, i = {}, 0
        for name in priors:
            out[name] = fixed[name] if name in fixed else x[i]; i += name not in fixed
        return out

    def neg_loglik(x):
        p = getpval(x)
        
        if cfg['fitrev'] or cfg['fitrep']:
            # Fit reversal and/or repetition curves
            return -getll_c(p)
        else:
            # Fit responses
            return -getll_r(p)
    
    def getll_c(pval):
        """Compute log-likelihood for reversal/repetition curves"""
        # Simulate reversal and repetition curves
        c, _ = sim_c(pval)
        ll = 0
        
        if cfg['fitrev']:  # fit reversal curve
            rrev_hat = np.mean(c['rrev'], axis=1)
            rrev_hat = (1 - lmin * 2) * rrev_hat + lmin
            ll += np.sum((csub['rrev'] * csub['nrev']) * np.log(rrev_hat))
            ll += np.sum(((1 - csub['rrev']) * csub['nrev']) * np.log(1 - rrev_hat))
        
        if cfg['fitrep']:  # fit repetition curve
            rrep_hat = np.mean(c['rrep'], axis=1)
            rrep_hat = (1 - lmin * 2) * rrep_hat + lmin
            ll += np.sum((csub['rrep'] * csub['nrep']) * np.log(rrep_hat))
            ll += np.sum(((1 - csub['rrep']) * csub['nrep']) * np.log(1 - rrep_hat))
        
        if cfg['fitcnf']:  # fit confidence
            if cfg['fitrev']:
                crev_hat = np.mean(c['crev'], axis=1)
                crev_hat = (1 - lmin * 2) * crev_hat + lmin
                ll += np.sum((csub['crev'] * csub['nrev']) * np.log(crev_hat))
                ll += np.sum(((1 - csub['crev']) * csub['nrev']) * np.log(1 - crev_hat))
            
            if cfg['fitrep']:
                crep_hat = np.mean(c['crep'], axis=1)
                crep_hat = (1 - lmin * 2) * crep_hat + lmin
                ll += np.sum((csub['crep'] * csub['nrep']) * np.log(crep_hat))
                ll += np.sum(((1 - csub['crep']) * csub['nrep']) * np.log(1 - crep_hat))
        
        return ll
    
    def getll_r(pval):
        """Compute log-likelihood for responses"""
        rbef, raft, cbef, caft, _ = sim_r(pval)
        ll = 0
        
        r_hat = np.mean(raft == 1, axis=1)
        r_hat = (1 - lmin * 2) * r_hat + lmin
        r_hat[cfg['raft'] == 2] = 1 - r_hat[cfg['raft'] == 2]
        ll += np.sum(np.log(r_hat))
        
        if cfg['fitcnf']:
            c_hat = np.mean(caft == 1, axis=1)
            c_hat = (1 - lmin * 2) * c_hat + lmin
            c_hat[cfg['caft'] == 2] = 1 - c_hat[cfg['caft'] == 2]
            ll += np.sum(np.log(c_hat))
        
        return ll
    
    def sim_c(pval):
        """Simulate responses and return behavioral curves"""
        rbef, raft, cbef, caft, psmp = sim_r(pval)
        c = getc(cfg['seqind'], cfg['seqpos'], cfg['seqdir'], cfg['seqllr'],
                 rbef, raft, cbef, caft)
        return c, psmp
    
    def sim_r(pval):
        """Simulate responses"""
        h = pval['h']
        sinf = pval['sinf']
        ssel = pval['ssel']
        scnf = pval.get('scnf', np.nan)
        tcnf = pval.get('tcnf', np.nan)
        gcnf = pval.get('gcnf', np.nan)
        
        # Initialize arrays
        xt = np.zeros((nseq, cfg['nsmp']))
        rbef = np.full((nseq, cfg['nsmp']), np.nan)
        raft = np.full((nseq, cfg['nsmp']), np.nan)
        cbef = np.full((nseq, cfg['nsmp']), np.nan)
        caft = np.full((nseq, cfg['nsmp']), np.nan)
        psmp = np.ones(nseq)
        
        for iseq in range(nseq):
            # Set state before sequence
            if cfg['seqind'][iseq] > 1:
                rbef[iseq, :] = raft[iseq - 1, :]
                if cfg['fitcnf']:
                    cbef[iseq, :] = caft[iseq - 1, :]
            else:  # 1st trial of each block
                rbef[iseq, :] = cfg['rbef'][iseq]
                if cfg['fitcnf']:
                    cbef[iseq, :] = cfg['cbef'][iseq]
            
            # Update log-belief
            if cfg['seqind'][iseq] > 1:
                xt[iseq, :] = upfun(xt[iseq - 1, :], h)
            
            xt[iseq, :] = np.random.normal(
                xt[iseq, :] + cfg['seqllr'][iseq],
                np.sqrt(cfg['seqlen'][iseq]) * sinf
            )
            
            # Apply selection noise
            xr = np.random.normal(xt[iseq, :], ssel)
            # Compute response
            raft[iseq, :] = 1 + (xr < 0).astype(int)
            
            if cfg['fitcnf']:
                # Compute log-belief in favor of response
                xc = xt[iseq, :] * (3 - 2 * raft[iseq, :])
                # Apply confidence gain during switches
                iswi = raft[iseq, :] != rbef[iseq, :]
                xc[iswi] = xc[iswi] * gcnf
                # Apply confidence noise
                xc = np.random.normal(xc, scnf)
                # Compute confidence
                caft[iseq, :] = 1 + (xc > tcnf).astype(int)
                
                # if cfg['resamp']:  # resample log-belief
                #     ires = ((raft[iseq, :] == cfg['raft'][iseq]) & 
                #            (caft[iseq, :] == cfg['caft'][iseq]))
                #     psmp[iseq] = np.mean(ires)
                #     if np.any(ires):
                #         # Bootstrap resample
                #         idx_ires = np.where(ires)[0]
                #         xt[iseq, :] = xt[iseq, np.random.choice(idx_ires, cfg['nsmp'], replace=True)]
                #     else:
                #         xt[iseq, :] = 0
            # else:
            #     if cfg['resamp']:  # resample log-belief
            #         ires = raft[iseq, :] == cfg['raft'][iseq]
            #         psmp[iseq] = np.mean(ires)
            #         if np.any(ires):
            #             # Bootstrap resample
            #             idx_ires = np.where(ires)[0]
            #             xt[iseq, :] = xt[iseq, np.random.choice(idx_ires, cfg['nsmp'], replace=True)]
            #         else:
            #             xt[iseq, :] = 0
        
        return rbef, raft, cbef, caft, psmp



    best_fval, best_x, best_result = np.inf, None, None
    options = {
        'uncertainty_handling': True,
        'noise_final_samples': cfg['nval'],
        'display': 'iter' if cfg['verbose'] >= 2 else ('final' if cfg['verbose'] == 1 else 'none'),
        'max_fun_evals': 500 * len(bounds['x0']),
        'max_iter': 200 * len(bounds['x0']),
        'nonlinear_scaling': True
    }

    for _ in range(cfg['nrun']):
        x0 = np.random.uniform(bounds['plb'], bounds['pub'])
        bads = BADS(
            neg_loglik,
            x0=x0,
            lower_bounds=bounds['lb'],
            upper_bounds=bounds['ub'],
            plausible_lower_bounds=bounds['plb'],
            plausible_upper_bounds=bounds['pub'],
            options=options
        )
        result = bads.optimize()
        if hasattr(result, 'success') and result.success:
            if result.fval < best_fval:
                best_x = result.x
                best_fval = result.fval
                best_result = result
       

    if best_x is None:
        best_x = np.array(bounds['x0'])
        best_fval = neg_loglik(best_x)
        best_result = None
        print("⚠️ Optimization failed. Using initial values.")

    phat = getpval(best_x)
    out = phat.copy()
    try:
        ll_sd = estimate_ll_sd(phat, cfg, nres=cfg['nres'])
        if ll_sd is None:
            ll_sd = 1e-6
    except Exception as e:
        print(f"Warning: estimate_ll_sd failed with error: {e}")
        ll_sd = 1e-6
    
    out.update({
        'xhat': best_x,
        'xnam': free_names,
        'll': -best_fval,
        'll_sd': ll_sd,
        'aic': -2 * (-best_fval) + 2 * len(best_x) + 2 * len(best_x) * (len(best_x) + 1) / (nseq - len(best_x) + 1),
        'bic': -2 * (-best_fval) + len(best_x) * np.log(nseq),
        'nsmp': cfg['nsmp'],
        'nval': cfg['nval'],
        'nrun': cfg['nrun'],
        'ntrl': nseq,
        'nfit': len(best_x),
        'cfg': cfg,
        'output': best_result
    })

    out['csub'] = csub

    # Store reversal and repetition curves with resamp disabled (like MATLAB line 271)
    resamp_orig = cfg['resamp']
    cfg['resamp'] = False
    out['cfit'], _ = sim_c(phat)
    cfg['resamp'] = resamp_orig
    
    if cfg['resamp']:
        # Store fraction of propagated samples
        _, out['psmp'] = sim_c(phat)

    return out


def upfun(x, h):
    """Update log-belief based on hazard rate h."""
    ratio = (1 - h) / h
    return x + np.log(ratio + np.exp(-x)) - np.log(ratio + np.exp(+x))
