
import numpy as np

import general.utility as u
import composite_tangling.code_creation as cc
import scipy.stats as sts

def get_model_performance(snrs, n_feats, n_values, n_neurs, metric=cc.hamming,
                          n=5000, n_boots=1000, noise_var=1,
                          model=cc.LinearCode, **model_args):
    perf = np.zeros((n_boots, len(snrs)))
    pwrs = noise_var*snrs**2
    for i, pwr in enumerate(pwrs):
        lc = model(n_feats, n_values, n_neurs, power=pwr, noise_cov=noise_var,
                   **model_args)
        trls = lc.compute_trl_metric(metric=metric, n=n)
        perf[:, i] = u.bootstrap_list(trls, np.nanmean, n=n_boots)
    return perf

def get_ccgp_dim_tradeoff_snr(snrs, n_trades, *args, **kwargs):
    ccgps = np.zeros((len(snrs), n_trades))
    shatts = np.zeros_like(ccgps)
    for i, snr in enumerate(snrs):
        out = get_ccgp_dim_tradeoff(snr, n_trades, *args, **kwargs)
        ccgps[i], shatts[i], max_d, trades = out
    return ccgps, shatts, max_d, trades

def get_lin_nonlin_pwr(snrs, trades, noise_var=1):
    pwrs = np.expand_dims(noise_var*snrs**2, 1)
    trades = np.expand_dims(trades, 0)
    lin_pwrs = trades*pwrs
    nonlin_pwrs = (1 - trades)*pwrs
    return lin_pwrs, nonlin_pwrs

def get_n_equal_partitions(n):
    if n % 2 > 0:
        n = n - 1
    num = ss.factorial(n)
    denom = 2*ss.factorial(n/2)**2
    return num/denom

def nonlin_class_performance(p_n, n_s, noise_var=1):
    n_c = get_n_equal_partitions(n_s)
    p_class = sts.norm().cdf(np.sqrt(p_n/noise_var))**n_s1
    return p_class

def partition_error_rate(p, noise_var, n_s):
    pe = sts.norm(0, 1).cdf(-np.sqrt(p/(n_s*noise_var)))
    return pe

def get_ccgp_dim_tradeoff(snr, n_trades, n_feats, n_values, n_neurs,
                          noise_var=1, codes=None):
    ccgps = np.zeros(n_trades)
    shatt = np.zeros(n_trades)
    pwr = noise_var*snr**2
    if codes is None:
        codes = [cc.LinearCode, cc.DiscreteMixedCode]
    trades = np.linspace(0, 1, n_trades)
    for i, t in enumerate(trades):
        code_props = [t, 1 - t]
        c = cc.CompositeCode(codes, code_props, n_feats, n_values, n_neurs,
                             total_power=pwr)

        sd = c.compute_shattering()
        shatt[i] = sd[0]
        max_d = sd[1]

        ccgp = c.compute_ccgp()
        ccgps[i] = np.mean(ccgp)

    return ccgps, shatt, max_d, trades
    
