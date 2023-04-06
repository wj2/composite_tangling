
import itertools as it
import scipy.stats as sts
import scipy.linalg as sla
import numpy as np
import sklearn.decomposition as skd
import sklearn.svm as skc
import scipy.special as ss

import mixedselectivity_theory.nms_discrete as nmd
import general.utility as u

def l2_norm(arr, axis=1):
    return np.sum(arr**2, axis=1)

def hamming(xs, ys):
    if len(xs.shape) > 2:
        axis = (1, 2)
    else:
        axis = 1
    b = np.all(xs == ys, axis=axis)
    return b

def hamming_set(stim1, stim2):
    same = np.zeros(len(stim1), dtype=bool)
    for i, s1_i in enumerate(stim1):
        set_1 = set(tuple(s1_ij) for s1_ij in s1_i)
        set_2 = set(tuple(s2_ij) for s2_ij in stim2[i])
        same[i] = set_1 == set_2
    return same

def mse(xs, ys):
    if len(xs.shape) > 2:
        axis = (1, 2)
    else:
        axis = 1
    m = np.sum((xs - ys)**2, axis=axis)
    return m

class Code(object):

    def __init__(self, n_feats, n_values, n_neurs, power=1, noise_cov=1,
                 pwr_func=nmd.empirical_variance_power_func, use_radius=True,
                 **stim_kwargs):
        if use_radius:
            pwr_func = nmd.empirical_radius_power_func
        self.n_feats = n_feats
        self.n_neurs = n_neurs
        self.n_values = n_values
        self.n_stimuli = n_values**n_feats
        self.power = power
        out = self.make_stimuli(n_feats, n_values, **stim_kwargs)
        self.stim, self.stim_proc, self.stim_dict = out
        self.stim_arr = np.array(self.stim)
        self.inv_stim_dict = {v:k for k, v in self.stim_dict.items()}
        self.noise_distr = sts.multivariate_normal(np.zeros(n_neurs), noise_cov)
        self.snr = np.sqrt(power/noise_cov)
        self.encoding_matrix = self.make_encoding_matrix(n_neurs, self.stim_proc,
                                                         power=power,
                                                         pwr_compute=pwr_func)
        out = self.make_representations(self.encoding_matrix, self.stim_proc)
        self.rep, self.rep_dict = out
        self.inv_rep_dict = {tuple(v):tuple(k) for k, v in self.rep_dict.items()}

    def make_encoding_matrix(self, n, stim, power=1, eps=10**-4,
                             pwr_compute=nmd.empirical_variance_power_func,
                             pwr_scale=nmd.l2_filt_scale, orth_basis=True):
        stim = np.array(stim)
        norm_var = np.var(stim, axis=0)
        mat = sts.multivariate_normal(0, 1).rvs((n, stim.shape[1]))
        if orth_basis:
            mat = sla.orth(mat)
        # mat_mean = np.expand_dims(np.mean(mat, axis=0), axis=0)
        # mat = mat - mat_mean
        # mat_var = np.expand_dims(np.var(mat, axis=0), axis=0)
        # mat = mat/mat_var
        enc_stim = np.dot(stim, mat.T)
        emp_pwr = pwr_compute(enc_stim)
        mat = mat*pwr_scale(power, emp_pwr)
        final_power = pwr_compute(np.dot(stim, mat.T))
        assert np.abs(final_power - power) < eps
        return mat

    def get_minimum_distance(self):
        stim1 = (0,)*self.n_feats
        stim2 = (1,) + (0,)*(self.n_feats - 1)
        r1 = self.get_representation(stim1)
        r2 = self.get_representation(stim2)
        d = np.sqrt(np.sum((r1 - r2)**2))
        return d

    def get_all_distances(self):
        dists_all = np.zeros((len(self.rep), len(self.rep)))
        for i, stim_i in enumerate(self.rep):
            dists_all[i] = u.euclidean_distance(stim_i, self.rep)
        return dists_all

    def get_min_distance_ns(self, eps=.001, second=False):
        md = self.get_minimum_distance()
        if second:
            md = np.sqrt(2)*md
        dists_all = self.get_all_distances()
        close = np.abs(dists_all - md) < eps
        ns = np.sum(close, axis=1)
        return ns

    def get_empirical_minimum_distance(self):
        min_dist = np.inf
        for (rep1, rep2) in it.combinations(self.rep, 2):
            d = np.sqrt(np.sum((rep1 - rep2)**2))
            min_dist = np.min((d, min_dist))
        return min_dist
    
    def make_representations(self, mat, stim):
        reps = np.dot(stim, mat.T)
        rep_dict = dict(zip(stim, reps))
        return reps, rep_dict

    def get_all_representations(self):
        return np.array(list(self.rep_dict.values()))
    
    def get_representation(self, stim, noise=False):
        stim = np.array(stim)
        if len(stim.shape) == 1:
            stim = np.expand_dims(stim, 0)
        if len(stim.shape) == 2:
            stim = np.expand_dims(stim, 1)
        reps = np.zeros((stim.shape[0], self.n_neurs))
        for i in range(stim.shape[1]):
            rep_i = np.array(list(self.rep_dict[self.stim_dict[tuple(s[i])]]
                                  for s in stim))
            reps = reps + rep_i
        if noise:
            reps = self._add_noise(reps)
        return reps

    def _add_noise(self, reps):
        noisy_reps = reps + self.noise_distr.rvs(reps.shape[0])
        return noisy_reps

    def _get_ccgp_stim_sets(self):
        f_combs = list(it.combinations(range(self.n_values), 2))
        train_sets = []
        test_sets = []
        for i in range(self.n_feats):
            for j, comb in enumerate(f_combs):
                train_stim_ind = np.random.choice(self.n_stimuli, 1)[0]
                train_stim = self.stim[train_stim_ind]

                c1_eg_stim = list(train_stim)
                c1_eg_stim[i] = comb[0]
                c1_eg_stim = tuple(c1_eg_stim)
                c1_eg_rep = self.rep_dict[self.stim_dict[c1_eg_stim]]
                c1_eg_rep = np.expand_dims(c1_eg_rep, 0)
                
                c2_eg_stim = list(train_stim)
                c2_eg_stim[i] = comb[1]
                c2_eg_stim = tuple(c2_eg_stim)
                c2_eg_rep = self.rep_dict[self.stim_dict[c2_eg_stim]]
                c2_eg_rep = np.expand_dims(c2_eg_rep, 0)

                stim_arr = np.array(self.stim)
                c1_test_stim = stim_arr[stim_arr[:, i] == comb[0]]
                c1_exclusion = np.any(c1_test_stim != c1_eg_stim, axis=1)
                c1_test_stim = c1_test_stim[c1_exclusion]
                c1_test_rep = self.get_representation(c1_test_stim)
                
                c2_test_stim = stim_arr[stim_arr[:, i] == comb[1]]
                c2_exclusion = np.any(c2_test_stim != c2_eg_stim, axis=1)
                c2_test_stim = c2_test_stim[c2_exclusion]
                c2_test_rep = self.get_representation(c2_test_stim)

                # print(c1_eg_stim) 
                # print(c1_test_stim)

                # print(c2_eg_stim)
                # print(c2_test_stim)
                train_sets.append((c1_eg_rep, c2_eg_rep))
                test_sets.append((c1_test_rep, c2_test_rep))
        return train_sets, test_sets                

    def compute_specific_ccgp(self, train_dim, gen_dim, train_dist=1,
                              gen_dist=1, n_reps=10, ref_stim=None,
                              train_noise=False, n_train=10, **dec_kwargs):
        if ref_stim is None:
            ref_stim = self.stim[0]
        tr_stim = tuple(rs + train_dist*(i == train_dim)
                        for i, rs in enumerate(ref_stim))
        gen_stim1 = tuple(rs + gen_dist*(i == gen_dim)
                          for i, rs in enumerate(ref_stim))
        gen_stim2 = tuple(rs + gen_dist*(i == gen_dim)
                          for i, rs in enumerate(tr_stim))
        tr_rep1 = self.get_representation(ref_stim)
        tr_rep2 = self.get_representation(tr_stim)
        te_rep1 = self.get_representation(gen_stim1)
        te_rep2 = self.get_representation(gen_stim2)
        pcorr = self.decode_rep_classes(tr_rep1, tr_rep2,
                                        c1_test=te_rep1,
                                        c2_test=te_rep2,
                                        n_reps=n_reps,
                                        train_noise=train_noise,
                                        n_train=n_train,
                                        **dec_kwargs)
        return pcorr
    
    def compute_ccgp(self, use_vals=1, n_reps=10, **dec_kwargs):
        train_sets, test_sets = self._get_ccgp_stim_sets()
        pcorr = np.zeros((len(train_sets), n_reps))
        for i, (c1_train, c2_train) in enumerate(train_sets):
            c1_test, c2_test = test_sets[i]
            pcorr[i] = self.decode_rep_classes(c1_train, c2_train,
                                               c1_test=c1_test,
                                               c2_test=c2_test,
                                               n_reps=n_reps,
                                               **dec_kwargs)
        return pcorr

    def _sample_noisy_reps(self, rs, n, add_noise=True, ret_stim=False,
                           ref_stim=None):
        r_inds = np.random.choice(rs.shape[0], int(n))
        reps = rs[r_inds]
        if add_noise:
            r_noisy_reps = self._add_noise(reps)
        else:
            r_noisy_reps = reps
        if ret_stim:
            out = (r_noisy_reps, rs)
        else:
            out = r_noisy_reps
        return out

    def sample_stim(self, n_samps=1000):
        inds = np.random.choice(self.n_stimuli, n_samps)
        return np.array(self.stim)[inds]
        
    def _make_decoding_reps(self, c1, c2, n, add_noise=True,
                            balance_training=False):
        n_half = int(n/2)
        c1_train_reps, stim = self._sample_noisy_reps(c1, n_half,
                                                      add_noise=add_noise,
                                                      ret_stim=True)
        if balance_training:
            ref_stim = stim
        else:
            ref_stim = None
        c2_train_reps = self._sample_noisy_reps(c2, n_half, add_noise=add_noise,
                                                ref_stim=ref_stim)
        reps = np.concatenate((c1_train_reps, c2_train_reps), axis=0)
        labels = np.concatenate((np.zeros(n_half), np.ones(n_half)))
        return reps, labels
    
    def decode_rep_classes(self, c1, c2, n_train=10**3, n_test=10**2,
                           n_reps=10, classifier=skc.SVC, kernel='linear',
                           train_noise=True, c1_test=None, c2_test=None,
                           test_noise=True, balance_training=False,
                           **classifier_params):
        pcorr = np.zeros(n_reps)
        for i in range(n_reps):
            out = self._make_decoding_reps(c1, c2, n_train,
                                           balance_training=balance_training,
                                           add_noise=train_noise)
            reps_train, labels_train = out
            if c1_test is None:
                c1_test = c1
            if c2_test is None:
                c2_test = c2
            out = self._make_decoding_reps(c1_test, c2_test,
                                           n_test, add_noise=test_noise)
            reps_test, labels_test = out 
            c = classifier(kernel=kernel, **classifier_params)
            c.fit(reps_train, labels_train)
            pcorr[i] = c.score(reps_test, labels_test)
        return pcorr

    def _get_partitions(self, random_thr=100, sample=None):
        if sample is None:
            sample = random_thr
        class_size = int(np.floor(self.n_stimuli/2))
        if ss.comb(self.n_stimuli, class_size) > random_thr:
            partitions = list(np.random.choice(self.n_stimuli,
                                               class_size,
                                               replace=False)
                              for x in range(sample))
        else:
            combs = it.combinations(range(self.n_stimuli), class_size)
            partitions = list(combs)
            n_parts = int(np.ceil(len(partitions)/2))
            partitions = partitions[:n_parts]
        return np.array(partitions)
    
    def compute_shattering(self, n_reps=5, thresh=.75, **dec_args):
        partitions = self._get_partitions()
        n_parts = len(partitions)
        pcorrs = np.zeros((n_parts, n_reps))
        rep_arr = np.array(self.rep)
        for i, ps1 in enumerate(partitions):
            ps2 = list(set(range(self.n_stimuli)).difference(ps1))
            c1 = rep_arr[ps1]
            c2 = rep_arr[ps2]
            pcorrs[i] = self.decode_rep_classes(c1, c2, n_reps=n_reps,
                                                **dec_args)
        n_c = np.sum(np.mean(pcorrs, axis=1) > thresh)
        n_dim = np.log2(2*n_c)
        n_dim_poss = np.log2(2*n_parts)
        return n_dim, n_dim_poss, pcorrs
            
    def compute_pca_dimensionality(self, v_explained=None, **pca_args):
        if v_explained is None:
            v_explained = 1 - 1/self.n_stimuli
        p = skd.PCA(n_components=v_explained, **pca_args)
        p.fit(self.rep)
        pev = p.explained_variance_ratio_
        n_dims = len(pev)
        out = (n_dims, pev, p)
        return out

    def get_stim_from_reps(self, reps):
        stim = np.zeros((reps.shape[0], self.n_feats))
        for i, r in enumerate(reps):
            stim[i] = self.inv_stim_dict[self.inv_rep_dict[tuple(r)]]
        return stim

    def sample_stimuli(self, n, n_stim=1, squeeze=True):
        rand_inds = np.random.choice(self.n_stimuli, n*n_stim)
        out = np.reshape(self.stim_arr[rand_inds], (n, n_stim, -1))
        if squeeze:
            out = np.squeeze(out)
        return out

    def sample_stim_reps(self, n, n_stim=1, noise=True):
        stim = self.sample_stimuli(n, n_stim=n_stim, squeeze=False)
        reps = self.get_representation(stim, noise=noise)
        return stim, reps

    def _get_rep_distances(self, reps, n_stim=1, dist_metric=l2_norm):
        stim_combs_mixes = np.array(list(it.combinations(self.stim,
                                                         n_stim)))
        stim_combs_doubles = np.stack((self.stim,)*n_stim, axis=1)
        stim_combs = np.concatenate((stim_combs_mixes, stim_combs_doubles),
                                    axis=0)
        out = np.zeros((len(reps), len(stim_combs)))
        for i, sc in enumerate(stim_combs):
            mean_rep = self.get_representation(np.expand_dims(sc, 0))
            rep_minus = reps - mean_rep
            out[:, i] = dist_metric(rep_minus)
        return stim_combs, out

    def get_average_swaps(self, n_stim, n=100, eps=.001):
        stim, reps = self.sample_stim_reps(n, n_stim=n_stim, noise=False)
        combs, dist = self._get_rep_distances(reps, n_stim=n_stim)
        print(stim[0])
        print(dist[0])
        print(dist[0] < eps)
        total = np.sum(dist < eps, axis=1) - 1
        mean = np.mean(total)
        return total, mean
    
    def get_posterior_probability(self, reps, n_stim=1):
        return self._get_rep_distances(reps, n_stim=n_stim,
                                       dist_metric=self.noise_distr.logpdf)
    
    def compute_trl_metric(self, metric=hamming, n=5000, n_stim=1):
        stim_orig, reps = self.sample_stim_reps(n, n_stim)
        stim_dec = self.decode_nn(reps, n_stim=n_stim)
        m = metric(np.squeeze(stim_orig), stim_dec)
        return m
        
    def compute_mse(self, **kwargs):
        return self.compute_trl_metric(metric=mse, **kwargs)

    def compute_discrim(self, n_stim=1, **kwargs):
        if n_stim > 1:
            metric = hamming_set
        else:
            metric = hamming
        return self.compute_trl_metric(metric=metric, n_stim=n_stim,
                                       **kwargs)

    def decode_nn(self, noisy_reps, n_stim=1):
        stim_combs, outs = self._get_rep_distances(noisy_reps, n_stim=n_stim)
        min_inds = np.argmin(outs, axis=1)
        stim_dec = stim_combs[min_inds]
        # stim_dec = np.zeros((noisy_reps.shape[0], self.n_feats))
        # for i, nr in enumerate(noisy_reps):
        #     rep, _ = nmd.decode_word(nr, self.rep)
        #     ps = self.inv_rep_dict[tuple(rep)]
        #     stim_dec[i] = self.inv_stim_dict[ps]
        return np.squeeze(stim_dec)
    
class LinearCode(Code):

    def make_stimuli(self, n_feats, n_values, symm=True):
        stim = list(it.product(range(n_values), repeat=n_feats))
        stim_proc = list(it.product(np.linspace(-1, 1, n_values),
                                    repeat=n_feats))
        stim_dict = dict(zip(stim, stim_proc))
        return stim, stim_proc, stim_dict
    
class DiscreteMixedCode(Code):

    def make_stimuli(self, n_feats, n_values, order=None):
        if order is None:
            order = n_feats
        out = nmd.generate_types((n_values,)*n_feats, order=order, excl=True)
        stim, binary_stim, transform, _ = out
        stim_proc = list(tuple(tbs) for tbs in transform(binary_stim))
        stim_dict = dict(zip(stim, stim_proc))
        return stim, stim_proc, stim_dict

def make_code(tradeoff, total_power, *args, codes=None, **kwargs):
    if codes is None:
        codes = [LinearCode, DiscreteMixedCode]
    code_props = [tradeoff, 1 - tradeoff]
    code = CompositeCode(codes, code_props, *args, total_power=total_power,
                         **kwargs)
    return code
    
class CompositeCode(Code):
    
    def __init__(self, codes, powers, n_feats, n_values, n_neurs,
                 total_power=None, noise_cov=1, stim_kwargs=None,
                 code_pwr=nmd.empirical_variance_power_func,
                 eps=.15):
        if stim_kwargs is None:
            stim_kwargs = ({},)*len(codes)
        if total_power is None:
            total_power = np.sum(powers)
        else:
            powers = total_power*np.array(powers)/np.sum(powers)
        code_list = []
        for i, code in enumerate(codes):
            c = code(n_feats, n_values, n_neurs, power=powers[i],
                     **stim_kwargs[i])
            code_list.append(c)
        self.codes = code_list
        super().__init__(n_feats, n_values, n_neurs, power=total_power,
                         noise_cov=noise_cov)
        
    def make_stimuli(self, n_feats, n_values):
        stim = self.codes[0].stim
        stim_proc = list(zip(*tuple(c.stim_proc
                                    for c in self.codes)))
        stim_dict = dict(zip(stim, stim_proc))
        return stim, stim_proc, stim_dict

    def make_encoding_matrix(self, n, stim, **kwargs):
        stim = np.array(stim, dtype=object)
        mat = np.zeros((n, stim.shape[1]))
        return mat
    
    def make_representations(self, mat, stim):
        full_rep = np.array(list(c.rep for c in self.codes))
        rep = np.sum(full_rep, axis=0)
        rep_dict = dict(zip(stim, rep))
        return rep, rep_dict
