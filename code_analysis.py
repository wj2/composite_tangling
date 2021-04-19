
import numpy as np
import functools as ft
import multiprocessing as mp
import itertools as it
import scipy.optimize as so

import general.utility as u
import composite_tangling.code_creation as cc
import scipy.stats as sts
import sklearn.svm as svm

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
    snrs = np.array(snrs)
    pwrs = np.expand_dims(noise_var*snrs**2, 1)
    trades = np.expand_dims(trades, 0)
    lin_pwrs = trades*pwrs
    nonlin_pwrs = (1 - trades)*pwrs
    return lin_pwrs, nonlin_pwrs

def partition_error_rate(p, noise_var, n_s):
    pe = sts.norm(0, 1).cdf(-np.sqrt(p/(n_s*noise_var)))
    return pe

def ang_to_err(xy, b, c, noise_var):
    t = xy*np.sin(np.pi/2 - np.arctan(b, c))
    errs = sts.norm(0, 1).cdf(t/np.sqrt(noise_var))
    return errs, t

def ccgp_pair_error_rate(d_nl, lg, lc1, lc2, noise_var, n_neurs, n=5000):
    dot_distr = sts.multivariate_normal(np.zeros_like(lg), 1/n_neurs,
                                        allow_singular=True)
    dot_distr = sts.multivariate_normal(np.zeros_like(lg), 0,
                                        allow_singular=True)

    a = np.sqrt(lg**2 + .5*d_nl**2 + np.sqrt(2)*d_nl*lg*dot_distr.rvs(n))

    ai_ft = np.sqrt(2)*lg*d_nl*dot_distr.rvs(n)/4
    ai_st = np.sqrt(2)*lc2*d_nl*dot_distr.rvs(n)/2
    a1 = (ai_ft + ai_st)/a + a
    a2 = -(ai_ft + ai_st)/a + a
    
    b = np.sqrt(3/2)*lg*d_nl*dot_distr.rvs(n)/(2*a)
    b2 = np.sqrt(3/2)*lg*d_nl*dot_distr.rvs(n)/(2*a)

    c = np.sqrt(lc1**2 + d_nl**2 + 2*lc1*d_nl*dot_distr.rvs(n))
    c2 = np.sqrt(lc2**2 + d_nl**2 + 2*lc2*d_nl*dot_distr.rvs(n))
    x1 = 2*a1*b/(c*np.sqrt(1 - (2*b/c)**2))
    x2 = 2*a2*b/(c*np.sqrt(1 - (2*b/c)**2))
    # x2 = a2*b/c

    # d = (lc1**2 + (lc1*d_nl + lc2*d_nl)*dot_distr.rvs(n))/c
    d = (.5*lc1**2 + np.sqrt(2)*lc2*d_nl*dot_distr.rvs(n))/c
    # print((c2/2)**2, b2**2)
    # print((c2/2)**2 -  b2**2)
    y1 = np.sqrt((c2/2)**2 - b2**2)
    y2 = -np.sqrt((c2/2)**2 - b2**2)
    # print(y1, d/2)
    y1 = d/2
    y2 = -d/2
    
    xy1 = x1 - y1
    xy2 = x2 - y2

    err1, t1 = ang_to_err(xy1, b, c, noise_var)
    err2, t2 = ang_to_err(xy2, b, c, noise_var)
    err1 = sts.norm(0, 1).cdf((-d - c/2)/np.sqrt(noise_var))
    err2 = sts.norm(0, 1).cdf((d - c/2)/np.sqrt(noise_var))
    err = np.mean((err1 + 1 - err2)/2, axis=0)
    return err, t1, t2    

def analytic_necessary_power(k, ni, noise_var, n_neurs, flex_target=.95,
                             ccgp_target=.95, i=1, j=1, include_flex=True,
                             include_ccgp=True):
    d_f = sts.norm(0, 1).ppf(1 - flex_target)
    d_g = sts.norm(0, 1).ppf(1 - ccgp_target)
    n_s = ni**k

    if include_flex:
        p_n = (d_f**2)*n_s*noise_var
    else:
        p_n = 0
    if include_ccgp:
        a = i*12/(k*(ni**2 - 1))
        b = 2*p_n
        c_num = (d_g**2)*(k**2)*((ni**2 - 1)**2)*noise_var
        c_den = (12**2)*((i*j - .5*i**2)**2)
        c = c_num/c_den
        p_l = (c*a + np.sqrt((c*a)**2 + 4*b*c))/2
    else:
        p_l = 0
    return p_l, p_n

def find_necessary_power(k, ni, noise_var, n_neurs, n=5000, target=.95,
                         tolerance=.01, include_ccgp=True, include_flex=True,
                         reg_weight=0, print_theory_diff=False):
    def opt_func(arg):
        p_l, p_n = arg
        gen = 1 - ccgp_error_rate(p_l, p_n, k, ni, noise_var,
                                  n_neurs, n=n)[0]
        flex = 1 - partition_error_rate(p_n, noise_var, ni**k)
        loss = (include_ccgp*np.abs(target - gen)
                + include_flex*np.abs(target - flex)
                + reg_weight*np.sqrt(p_l + p_n))
        return loss
    res = so.minimize(opt_func, (10, 10), bounds=((0, None), (0, None)))
    p_l, p_n = res.x
    p_l_ana, p_n_ana = analytic_necessary_power(k, ni, noise_var, n_neurs,
                                                include_flex=include_flex,
                                                include_ccgp=include_ccgp)
    if print_theory_diff and res.fun > .0001:
        print('opt', p_l, p_n)
        print('ana', p_l_ana, p_n_ana)
        print('fun', res.fun)
    if print_theory_diff and (np.abs(p_l - p_l_ana) > .01
                              or np.abs(p_n - p_n_ana) > .01):
        print('--- off ---')
        print('opt', p_l, p_n)
        print('ana', p_l_ana, p_n_ana)
        print('fun', res.fun)
        print('-----------')
        
    if res.fun - reg_weight*np.sqrt(p_l + p_n) > tolerance:
        p_l = np.nan
        p_n = np.nan
    return p_l, p_n

def make_gen_flex_pwr_regions(reg_range, feat_range, val_range, *args,
                              **kwargs):
    out_npwr = np.zeros((len(reg_range), len(feat_range), len(val_range)))
    out_lpwr = np.zeros_like(out_npwr)
    for i, reg in enumerate(reg_range):
        out = make_gen_flex_pwr(feat_range, val_range, *args, regions=reg,
                                **kwargs)
        out_npwr[i], out_lpwr[i] = out
    return out_lpwr, out_npwr

def make_gen_flex_pwr(feat_range, val_range, noise_var, n_neurs, n=5000,
                      use_analytic=False, regions=1, assignment=True,
                      **kwargs):
    opt_npwr = np.zeros((len(feat_range), len(val_range)))
    opt_lpwr = np.zeros_like(opt_npwr)
    k_reg = np.array(feat_range)/regions
    k_reg[k_reg < 1] = 1
    if assignment and regions > 1:
        k_reg = k_reg + 1
    for i, k in enumerate(k_reg):
        for j, ni in enumerate(val_range):
            if use_analytic:
                p_lopt, p_nopt = analytic_necessary_power(k, ni, noise_var,
                                                          n_neurs, **kwargs)
            else:
                p_lopt, p_nopt = find_necessary_power(k, ni, noise_var,
                                                      n_neurs, n=n, **kwargs)
            opt_npwr[i, j] = regions*p_nopt
            opt_lpwr[i, j] = regions*p_lopt
    return opt_lpwr, opt_npwr

def make_ccgp_shatter_phase(pwr, n_trades, feat_range, val_range, noise_var,
                            n_neurs, n=5000):
    trades = np.linspace(0, 1, n_trades)
    combs = list(it.product(trades, feat_range, val_range))
    combs_inds = list(it.product(range(n_trades), range(len(feat_range)),
                                 range(len(val_range))))

    ccgp = np.zeros((n_trades, len(feat_range), len(val_range)))
    shatt = np.zeros_like(ccgp)
    
    for i, (t, k, ni) in enumerate(combs):
        p_l = pwr*(1 - t)
        p_n = pwr*t
        ccgp[combs_inds[i]] = ccgp_error_rate(p_l, p_n, k, ni, noise_var,
                                              n_neurs, n=n)[0]
        shatt[combs_inds[i]] = partition_error_rate(p_n, noise_var, ni**k)
        
    return trades, ccgp, shatt

def ccgp_error_rate(p_l, p_n, n_feats, n_values, noise_var, n_neurs, n=5000):
    d_nl = get_mixed_theory_distance(p_n, n_feats, n_values)
    d_lin = get_linear_theory_distance(p_l, n_feats, n_values)
    lg = d_lin
    lc1 = d_lin
    lc2 = d_lin
    # for 
    return ccgp_pair_error_rate(d_nl, lg, lc1, lc2, noise_var, n_neurs, n=n)

def get_code_properties(code, pair1, pair2):
    dmc, lc = code.codes
    pl1 = lc.get_representation(pair1)
    pl1 = pl1 - pl1[0]
    pl2 = lc.get_representation(pair2) - pl1[0]
    f1 = code.get_representation(pair1) - pl1[0]
    f2 = code.get_representation(pair2) - pl1[0]

    c_pr = np.sqrt(np.sum((f1[0] - f1[1])**2))

    c_f = svm.SVC(kernel='linear', C=1000)
    c_f.fit(f1, [0, 1])

    c_both = svm.SVC(kernel='linear', C=1000)
    f_comb = np.concatenate((f1, f2), axis=0)
    c_both.fit(f_comb, [0, 1, 0, 1])

    c_p = svm.SVC(kernel='linear', C=1000)
    c_p.fit(pl1, [0, 1])
    o_vec = u.generate_orthonormal_vectors(c_both.coef_[0], 1)
    dev_h_i = np.dot(o_vec, f1[0])
    dev_h_j = np.dot(o_vec, f1[1])
    dev_h = dev_h_i - dev_h_j

    f1_cent = np.mean(f1, axis=0)
    f2_cent = np.mean(f2, axis=0)
    
    cent_ax = u.make_unit_vector(f2_cent - f1_cent)
    vert_ax = u.make_unit_vector(f1[1] - f1[0])

    y_opp = np.dot(vert_ax, f2.T)
    y_cent = np.dot(vert_ax, f1_cent)
    y = y_opp # - y_cent
    
    sp = np.dot(cent_ax, f2.T) - np.dot(cent_ax, f1_cent)
    
    theta_svm = u.vector_angle(c_f.coef_[0], cent_ax, degrees=False) - np.pi/2
    t_svm = c_f.decision_function(f2/np.sqrt(np.sum(c_f.coef_**2)))
    t_svm = t_svm*c_pr
    x = sp[0]*np.tan(theta_svm)

    
    ax1 = u.make_unit_vector(pl1[1] - pl1[0])
    ax2 = u.make_unit_vector(pl2[0] - pl1[0])
    
    ax1 = u.make_unit_vector(f1[1] - f1[0])
    ax2 = u.make_unit_vector(f2[0] - f1[0])

    trs = np.stack((ax1, ax2), axis=0)
    trs_func = lambda x: np.dot(x - pl1[0], trs.T)
    trs_func = lambda x: np.dot(x - f1[0], trs.T)

    f1_p = trs_func(f1)
    f2_p = trs_func(f2)
    p1_p = trs_func(pl1)
    p2_p = trs_func(pl2)
    
    svs = (c_f.coef_[0], c_p.coef_[0])
    ints = (c_f.intercept_, c_p.intercept_)
    
    bp = np.mean(f1_p, axis=0) - np.mean(p1_p, axis=0)
    # sp = f2_p - np.mean(f1_p, axis=0)
    dev = (f1_p - p1_p)
    dev = np.sqrt(np.sum(np.diff(f1, axis=0)**2, axis=1))

    dev = np.sqrt(np.sum((np.mean(f1, axis=0) - np.mean(f2, axis=0))**2))
    
    v1 = np.diff(p1_p, axis=0)
    v2 = np.diff(f1_p, axis=0)
    theta1 = u.signed_vector_angle_2d(v1[0], v2[0], degrees=False)
    # theta_svm = u.vector_angle(svs[1], svs[0], degrees=False)
    tp_svm = c_p.decision_function(pl2 - pl1[0])
    # print(tp_svm)
    return c_pr, dev_h, y, sp, x, svs, ints, theta_svm, t_svm

def get_linear_code_distance(*args, **kwargs):
    return get_code_distance(*args, cc.LinearCode, **kwargs)

def get_mixed_code_distance(*args, **kwargs):
    return get_code_distance(*args, cc.DiscreteMixedCode, **kwargs)

def get_code_distance(pwrs, n_feats, n_values, code, n_neurs=100, **kwargs):
    ds = np.zeros(len(pwrs))
    for i, pwr in enumerate(pwrs):
        c = code(n_feats, n_values, n_neurs, power=pwr, **kwargs)
        ds[i] = c.get_minimum_distance()
    return ds

def get_linear_theory_distance(pwrs, n_feats, n_values, default_d=1):
    ds = np.sqrt(12*pwrs/(n_feats*(n_values**2 - 1)))
    return ds

def get_mixed_theory_distance(pwrs, n_feats, n_values):
    return np.sqrt(2*pwrs)

def _compute_metrics(codes, n_feats, n_values, n_neurs, trade,
                     compute_shattering=True, compute_discrim=True,
                     compute_mse=True, trade_reps=10, total_power=10,
                     noise_cov=1, **compute_kwargs):
    code_props = [trade, 1 - trade]
    total_dist = np.zeros(trade_reps)
    indiv_dists = np.zeros((trade_reps, len(codes)))
    shatt = None
    max_d = None
    p_perfs = None
    discrim_errs = None
    mse_errs = None
    for j in range(trade_reps):
        c = cc.CompositeCode(codes, code_props, n_feats, n_values, n_neurs,
                             total_power=total_power, noise_cov=noise_cov)
        total_dist[j] = c.get_minimum_distance()
        indiv_dists[j] = list(ci.get_minimum_distance() for ci in c.codes)
        if compute_shattering:
            sd = c.compute_shattering(**compute_kwargs)
            max_d = sd[1]
            p_scores = sd[2]
            if j == 0:
                p_perfs = np.zeros((trade_reps,)
                                   + p_scores.shape)
                shatt = np.zeros((trade_reps,)
                                 + sd[0].shape)
            p_perfs[j] = p_scores
            shatt[j] = sd[0]

        if compute_discrim:
            discrim_err = c.compute_discrim()
            if j == 0:
                discrim_errs = np.zeros((trade_reps,)
                                        + discrim_err.shape)
            discrim_errs[j] = discrim_err

        if compute_mse:
            mse_err = c.compute_mse()
            if j== 0:
                mse_errs = np.zeros((trade_reps,)
                                    + mse_err.shape)
            mse_errs[j] = mse_err

        ccgp = c.compute_ccgp(**compute_kwargs)
        if j == 0:
            ccgps = np.zeros((trade_reps,) + ccgp.shape)
        ccgps[j] = ccgp
    out = (ccgps, total_dist, indiv_dists, shatt, max_d, p_perfs, discrim_errs,
           mse_errs)
    return out
        
def _merge_parallel_output(out, compute_shattering, compute_discrim,
                           compute_mse):
    ccgps = []
    total_dist = []
    indiv_dists = []
    p_perfs = []
    discrim_errs = []
    mse_errs = []
    for o in out:
        ccgp, td, id_, shatt, max_d, p_perf, discrim_err, mse_err = o
        ccgps.append(ccgp)
        total_dist.append(td)
        indiv_dists.append(id_)
        p_perfs.append(p_perf)
        discrim_errs.append(discrim_err)
        mse_errs.append(mse_err)
    metrics = {'ccgp':np.array(ccgps)}
    info = {'min_dist_full':np.array(total_dist),
            'min_dist_code':np.array(indiv_dists)}
    if compute_shattering:
        metrics.update((('shattering',np.array(p_perfs)),))
        info.update((('shatt', shatt), ('max_d', max_d)))
    if compute_discrim:
        metrics.update((('discrim',np.array(discrim_errs)),))
    if compute_mse:
        metrics.update((('mse',np.array(mse_errs)),))
    return metrics, info

def get_ccgp_dim_tradeoff(snr, n_trades, n_feats, n_values, n_neurs, eps=10**-2,
                          noise_var=1, codes=None, compute_shattering=True,
                          compute_discrim=True, compute_mse=True, trade_reps=10,
                          **compute_kwargs):
    pwr = noise_var*snr**2
    if codes is None:
        codes = [cc.LinearCode, cc.DiscreteMixedCode]
    trades = np.linspace(eps, 1 - eps, n_trades)
    total_dist = np.zeros((len(trades), trade_reps))
    indiv_dists = np.zeros(trades.shape + (trade_reps, len(codes),))

    cm = ft.partial(_compute_metrics, codes, n_feats, n_values, n_neurs,
                    total_power=pwr, noise_cov=noise_var,
                    compute_mse=compute_mse, trade_reps=trade_reps,
                    compute_shattering=compute_shattering,
                    compute_discrim=compute_discrim, **compute_kwargs)
    with mp.Pool() as p:
        out = p.map(cm, trades)
    metrics, info = _merge_parallel_output(out, compute_shattering,
                                           compute_discrim, compute_mse)
    return trades, metrics, info
    
def get_multiple_tradeoffs(snrs, n_trades, n_feats, n_values, n_neurs,
                           trade_reps=10, n_reps=2, eps=0, train_noise=False,
                           noise_var=1, **compute_kwargs):
    arg_combs = it.product(snrs, n_feats, n_values, n_neurs)
    full_dict = {}
    for (snr, n_feat, n_val, n_neur) in arg_combs:
        out = get_ccgp_dim_tradeoff(snr, n_trades, n_feat, n_val,
                                    n_neur, trade_reps=trade_reps,
                                    n_reps=n_reps, train_noise=train_noise,
                                    noise_var=noise_var, eps=eps,
                                    **compute_kwargs)
        trades, metrics, info = out
        ids = info['min_dist_code']
        p_l, p_n = get_lin_nonlin_pwr([snr], trades, noise_var=noise_var)
        
        errs, t1, t2 = ccgp_error_rate(p_l, p_n, n_feats, n_values,
                                       noise_var, n_neur, n=10000)
        shatt_err = partition_error_rate(p_n, noise_var, n_val**n_feat)

        theor = {'ccgp':1 - errs, 'shattering':1 - shatt_err}

        out_dict = dict(metrics=metrics, info=info, theory=theor)
        full_dict[(snr, n_feat, n_val)] = out_dict
    return trades, full_dict
