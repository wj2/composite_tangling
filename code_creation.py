
import itertools as it
import scipy.stats as sts
import numpy as np
import sklearn.decomposition as skd
import sklearn.svm as skc

import mixedselectivity_theory.nms_discrete as nmd

def hamming(xs, ys):
    b = np.all(xs == ys, axis=1)
    return b

def mse(xs, ys):
    m = np.sum((xs - ys)**2, axis=1)
    return m

class Code(object):

    def __init__(self, n_feats, n_values, n_neurs, power=1, noise_cov=1,
                 **stim_kwargs):
        self.n_feats = n_feats
        self.n_neurs = n_neurs
        self.n_values = n_values
        self.n_stimuli = n_values**n_feats
        self.power = power
        out = self.make_stimuli(n_feats, n_values, **stim_kwargs)
        self.stim, self.stim_proc, self.stim_dict = out
        self.inv_stim_dict = {v:k for k, v in self.stim_dict.items()}
        self.noise_distr = sts.multivariate_normal(np.zeros(n_neurs), noise_cov)
        self.snr = np.sqrt(power/noise_cov)
        self.encoding_matrix = self.make_encoding_matrix(n_neurs, self.stim_proc,
                                                         power=power)
        out = self.make_representations(self.encoding_matrix, self.stim_proc)
        self.rep, self.rep_dict = out
        self.inv_rep_dict = {tuple(v):tuple(k) for k, v in self.rep_dict.items()}

    def make_encoding_matrix(self, n, stim, power=1, eps=10**-4,
                             pwr_compute=nmd.empirical_variance_power_func,
                             pwr_scale=nmd.l2_filt_scale):
        stim = np.array(stim)
        norm_var = np.var(stim, axis=0)
        mat = sts.multivariate_normal(0, 1).rvs((n, stim.shape[1]))
        mat_mean = np.expand_dims(np.mean(mat, axis=0), axis=0)
        mat = mat - mat_mean
        mat_var = np.expand_dims(np.var(mat, axis=0), axis=0)
        mat = mat/mat_var
        enc_stim = np.dot(stim, mat.T)
        emp_pwr = pwr_compute(enc_stim)
        mat = mat*pwr_scale(power, emp_pwr)
        final_power = pwr_compute(np.dot(stim, mat.T))
        assert np.abs(final_power - power) < eps
        return mat

    def make_representations(self, mat, stim):
        reps = np.dot(stim, mat.T)
        rep_dict = dict(zip(stim, reps))
        return reps, rep_dict
        
    def get_representation(self, stim, noise=False):
        if len(stim.shape) == 1:
            stim = np.expand_dims(stim, 0)
        reps = np.array(list(self.rep_dict[self.stim_dict[tuple(s)]]
                             for s in stim))
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

                train_sets.append((c1_eg_rep, c2_eg_rep))
                test_sets.append((c1_test_rep, c2_test_rep))
        return train_sets, test_sets                
    
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

    def _sample_noisy_reps(self, rs, n):
        r_inds = np.random.choice(rs.shape[0], int(n))
        r_noisy_reps = self._add_noise(rs[r_inds])
        return r_noisy_reps

    def _make_decoding_reps(self, c1, c2, n):
        n_half = int(n/2)
        c1_train_reps = self._sample_noisy_reps(c1, n_half)
        c2_train_reps = self._sample_noisy_reps(c2, n_half)
        reps = np.concatenate((c1_train_reps, c2_train_reps), axis=0)
        labels = np.concatenate((np.zeros(n_half), np.ones(n_half)))
        return reps, labels
    
    def decode_rep_classes(self, c1, c2, n_train=10**3, n_test=10**2,
                           n_reps=10, classifier=skc.SVC, kernel='linear',
                           c1_test=None, c2_test=None, **classifier_params):
        pcorr = np.zeros(n_reps)
        for i in range(n_reps):
            reps_train, labels_train = self._make_decoding_reps(c1, c2, n_train)
            if c1_test is None:
                c1_test = c1
            if c2_test is None:
                c2_test = c2
            reps_test, labels_test = self._make_decoding_reps(c1_test, c2_test,
                                                              n_test)
            c = classifier(kernel=kernel, **classifier_params)
            c.fit(reps_train, labels_train)
            pcorr[i] = c.score(reps_test, labels_test)
        return pcorr

    def _get_partitions(self):
        class_size = int(np.floor(self.n_stimuli/2))
        classes = (0,)*class_size + (1,)*class_size
        if class_size*2 < self.n_stimuli:
            classes = classes + (-1,)
        # this will become slow for large permutation sets
        partitions = np.unique(list(it.permutations(classes)), axis=0)
        return partitions
    
    def compute_shattering(self, n_reps=10, thresh=.75, **dec_args):
        partitions = self._get_partitions()
        n_parts = int(np.ceil(len(partitions)/2))
        pcorrs = np.zeros((n_parts, n_reps))
        for i, ps in enumerate(partitions[:n_parts]):
            c1 = self.rep[ps == 0]
            c2 = self.rep[ps == 1]
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
    
    def compute_trl_metric(self, metric=hamming, n=5000):
        rand_inds = np.random.choice(self.n_stimuli, n)
        reps = self.rep[rand_inds]
        noisy_reps = self._add_noise(reps)
        stim_orig = self.get_stim_from_reps(reps)
        stim_dec = self.decode_nn(noisy_reps)
        m = metric(stim_orig, stim_dec)
        return m
        
    def compute_mse(self, **kwargs):
        return self.compute_trl_metric(metric=mse, **kwargs)

    def compute_discrim(self, **kwargs):
        return self.compute_trl_metric(metric=hamming, **kwargs)

    def decode_nn(self, noisy_reps):
        stim_dec = np.zeros((noisy_reps.shape[0], self.n_feats))
        for i, nr in enumerate(noisy_reps):
            rep, _ = nmd.decode_word(nr, self.rep)
            ps = self.inv_rep_dict[tuple(rep)]
            stim_dec[i] = self.inv_stim_dict[ps]
        return stim_dec
    
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
        assert np.abs(code_pwr(self.rep) - self.power) < self.power*eps

    def make_stimuli(self, n_feats, n_values):
        stim = self.codes[0].stim
        stim_proc = list(zip(*tuple(c.stim_proc
                                    for c in self.codes)))
        stim_dict = dict(zip(stim, stim_proc))
        return stim, stim_proc, stim_dict

    def make_encoding_matrix(self, n, stim, **kwargs):
        stim = np.array(stim)
        mat = np.zeros((n, stim.shape[1]))
        return mat
    
    def make_representations(self, mat, stim):
        full_rep = np.array(list(c.rep for c in self.codes))
        rep = np.sum(full_rep, axis=0)
        rep_dict = dict(zip(stim, rep))
        return rep, rep_dict
