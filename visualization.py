
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skc
import scipy.stats as sts

import general.plotting as gpl
import general.utility as u
import composite_tangling.code_analysis as ca

from mpl_toolkits.mplot3d import Axes3D

def plot_script_results(trades, full_dict, ax=None,
                        x_label='linear/nonlinear ratio', **kwargs):
    outs = []
    for (snr, n_feats, n_values), out in full_dict.items():
        metrics, info, theory = out['metrics'], out['info'], out['theory']
        out = plot_metrics(trades, metrics, x_label, conf95=True, theory=theory,
                           **kwargs)
        outs.append(out)
    return outs

def plot_metrics(x_vals, metrics, x_label, y_labels=None, axs=None, fwid=3,
                 theory=None, eps=.1, theory_col='k', **kwargs):
    if theory is None:
        theory = {}
    if axs is None:
        fsize = (fwid*len(metrics), fwid)
        f, axs = plt.subplots(1, len(metrics), figsize=fsize)
        out = (f, axs)
    else:
        out = axs
    for i, (k, v) in enumerate(metrics.items()):
        ax = axs[i]
        if len(v.shape) == 3:
            v = np.expand_dims(v, 2)
        vs = v.shape
        col = None
        for j in range(vs[2]):
            v_plot = v[:, :, j]
            v_plot_shape = np.reshape(v_plot, (vs[0], vs[1]*vs[3]))
            
            l = gpl.plot_trace_werr(x_vals, v_plot_shape.T, ax=ax,
                                    color=col, **kwargs)
            col = l[0].get_color()
        
        v_theor = theory.get(k)
        if v_theor is not None:
            gpl.plot_trace_werr(x_vals, v_theor, linestyle='dashed',
                                color=theory_col, ax=ax)
        ax.set_title(k)
        ax.set_xlabel(x_label)
        if y_labels is not None:
            ax.set_ylabel(y_labels[k])
        yl = ax.get_ylim()
        if np.diff(yl) < eps:
            ax.set_ylim(yl[0] - eps, yl[1] + eps)
    return out

def _plot_pt(pt, ax, *args, **kwargs):
    ax.plot([pt[0]], [pt[1]], [pt[2]], *args, **kwargs)

def _plot_line(pt1, pt2, ax, *args, **kwargs):
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
            *args, **kwargs)

def _compute_theta(lg, d_nl, n_neurs, lc=None, n=10000):
    if lc is None:
        lc = lg    
    dot_distr = sts.multivariate_normal(np.zeros_like(lg), 1/n_neurs,
                                        allow_singular=True)
    a = np.sqrt(lg**2 + .5*d_nl**2 + np.sqrt(2)*d_nl*lg*dot_distr.rvs(n))
    b = np.sqrt(3/2)*lg*d_nl*dot_distr.rvs(n)/(2*a)
    c = np.sqrt(lc**2 + d_nl**2 + np.sqrt(2)*lc*d_nl*dot_distr.rvs(n))
    theta_theor = np.arcsin(2*b/c)
    return theta_theor

def plot_theta_var(lg, d_nls, n_neurs, lc=None, ax=None, n=5000,
                   label=''):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ts = np.zeros((n, len(n_neurs), len(d_nls)))
    for j, n_neur in enumerate(n_neurs):
        for i, d_nl in enumerate(d_nls):
            ts[:, j, i] = _compute_theta(lg, d_nl, n_neur, lc=lc, n=n)

    t_var = np.std(ts, axis=0)

    for j, n_neur in enumerate(n_neurs):
        gpl.plot_trace_werr(d_nls, t_var[j], ax=ax,
                            label='D = {}'.format(n_neur))
    ax.set_xlabel('nonlinear distance')
    ax.set_ylabel('angle standard deviation')
    return ax

def plot_high_d(pt1, pt2, pt3, ax=None, alpha=.35, alpha_pt=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    if alpha_pt is None:
        alpha_pt = alpha

    zp = [0, 0, 0]
    _plot_line(zp, pt1, ax, 'k', alpha=alpha)
    _plot_line(zp, pt2, ax, 'k', alpha=alpha)
    _plot_line(zp, pt3, ax, 'k', alpha=alpha)
    _plot_line(pt1, pt2, ax, 'k', alpha=alpha)
    _plot_line(pt3, pt2, ax, 'k', alpha=alpha)
    _plot_line(pt1, pt3, ax, 'k', alpha=alpha)

    _plot_pt(pt1, ax, 'ko', alpha=alpha_pt)
    _plot_pt(pt2, ax, 'ko', alpha=alpha_pt)
    _plot_pt(pt3, ax, 'ko', alpha=alpha_pt)

    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.set_zlim([0, 1.2])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.view_init(30, 20)
    ax.set_xlabel('neuron 1')
    ax.set_ylabel('neuron 2')
    ax.set_zlabel('neuron 3')   
    
def plot_schematic(pt1, f1, f2, d_l, d_nls, alpha=.35, ax=None,
                   vs=None, ortho_d=5, nl_color=(.2, .2, .8),
                   alpha_pt=None):
    if alpha_pt is None:
        alpha_pt = alpha
    pt1 = np.array(pt1)
    f1 = u.make_unit_vector(f1)
    f2 = u.make_unit_vector(f2)

    pt2 = pt1 + d_l*f1

    pt3 = pt1 + d_l*f2
    pt4 = pt1 + d_l*f2 + d_l*f1

    if vs is None:
        vs = u.generate_orthonormal_basis(ortho_d)
    v1 = vs[0, :3]
    v2 = vs[1, :3]
    v3 = vs[2, :3]
    v4 = vs[3, :3]
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')

    _plot_line(pt1, pt2, ax, 'k', alpha=alpha)
    _plot_line(pt3, pt4, ax, 'k', alpha=alpha)
    _plot_line(pt1, pt3, ax, 'k', alpha=alpha)
    _plot_line(pt2, pt4, ax, 'k', alpha=alpha)

    _plot_pt(pt1, ax, 'ko', alpha=alpha_pt)
    _plot_pt(pt2, ax, 'ko', alpha=alpha_pt)
    _plot_pt(pt3, ax, 'ko', alpha=alpha_pt)
    _plot_pt(pt4, ax, 'ko', alpha=alpha_pt)

    for d_nl in d_nls:
        pt1_f = pt1 + d_nl*v1
        pt2_f = pt2 + d_nl*v2
        pt3_f = pt3 + d_nl*v3
        pt4_f = pt4 + d_nl*v4

        _plot_line(pt1, pt1_f, ax, color=nl_color)
        _plot_line(pt2, pt2_f, ax, color=nl_color)
        _plot_line(pt1_f, pt2_f, ax, 'k')
    
        _plot_line(pt3, pt3_f, ax, color=nl_color)
        _plot_line(pt4, pt4_f, ax, color=nl_color)
        _plot_line(pt3_f, pt4_f, ax, 'k')
        _plot_line(pt1_f, pt3_f, ax, 'k')
        _plot_line(pt2_f, pt4_f, ax, 'k')

        _plot_pt(pt1_f, ax, 'ko')
        _plot_pt(pt2_f, ax, 'ko')
        _plot_pt(pt3_f, ax, 'ko')
        _plot_pt(pt4_f, ax, 'ko')

        m12 = (pt1_f + pt2_f)/2
        m34 = (pt3_f + pt4_f)/2
        _plot_line(m12, m34, ax)

        vax = u.make_unit_vector(pt2_f - pt1_f)
    
    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.25, 1.25])
    ax.set_zlim([-.25, 1.25])
    ax.set_xticks([-.5, 0, .5])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])

    ax.view_init(30, 20)
    ax.set_xlabel('neuron 1')
    ax.set_ylabel('neuron 2')
    ax.set_zlabel('neuron 3')   

def plot_decoder_surface(cent_dist, noise_var=1, cent_weights=(2,), ax=None,
                         n=1000, ax_color='k', samp_pts=1000, kernel='linear',
                         dec_dim=4, set_equal_aspect=True, modu=0,
                         **svm_kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if set_equal_aspect:
        ax.set_aspect('equal')
    s1 = np.array([cent_dist, 0])
    s2 = np.array([0, cent_dist])
    si = np.array([0, 0])

    circ_pts = gpl.gen_circle_pts(n, r=np.sqrt(noise_var))

    ax.plot([si[0], s1[0]], [si[1], s1[1]], color=ax_color)
    l1 = ax.plot(*s1, 'o')
    col1 = l1[0].get_color()
    ax.plot(s1[0] + circ_pts[:, 0], s1[1] + circ_pts[:, 1], color=col1)
    
    ax.plot([si[0], s2[0]], [si[1], s2[1]], color=ax_color)
    l2 = ax.plot(*s2, 'o')
    col2 = l2[0].get_color()
    ax.plot(s2[0] + circ_pts[:, 0], s2[1] + circ_pts[:, 1], color=col2)
    
    li = ax.plot(*si, 'o')
    coli = li[0].get_color()
    ax.plot(si[0] + circ_pts[:, 0], si[1] + circ_pts[:, 1], color=coli)

    noise_distr = sts.multivariate_normal(np.zeros(dec_dim), noise_var)
    stims = np.identity(dec_dim)*cent_dist
    n_each = int(np.floor(dec_dim/2))
    class1_cents = stims[:n_each]
    class2_cents = stims[n_each:2*n_each + 1]

    class1_pts = np.zeros((samp_pts*n_each, dec_dim))
    class2_pts = np.zeros_like(class1_pts)
    for i in range(n_each):
        cw1_i = stims[i:i+1] + noise_distr.rvs(samp_pts)
        class1_pts[i*samp_pts:(i + 1)*samp_pts, :] = cw1_i
        
        cw2_i = stims[n_each + i: n_each + i+1] + noise_distr.rvs(samp_pts)
        class2_pts[i*samp_pts:(i+1)*samp_pts, :] = cw2_i
    labels1 = np.zeros(class1_pts.shape[0])
    labels2 = np.ones_like(labels1)

    pts_all = np.concatenate((class1_pts, class2_pts), axis=0)
    labels_all = np.concatenate((labels1, labels2), axis=0)
    c = skc.SVC(kernel=kernel, **svm_kwargs)
    c.fit(pts_all, labels_all)
    p_cw = c.score(pts_all, labels_all)
    n_s = stims.shape[0]
    print(1 - ca.partition_error_rate(cent_dist**2, noise_var, n_s))

    xx = np.linspace(-np.sqrt(noise_var), cent_dist, 30)
    yy = xx
    YY, XX = np.meshgrid(yy, xx)
    mesh_pts = [XX.ravel(), YY.ravel()]
    mesh_pts_add = [modu*np.ones(mesh_pts[0].shape[0])]*(dec_dim - 2)
    xy = np.vstack(mesh_pts + mesh_pts_add).T
    Z = c.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # sv = c.coef_
    # off = c.intercept_
    # print(c.intercept_)
    # df_pts = np.array([-np.sqrt(noise_var), cent_dist])
    # y_pts = (-sv[0, 0]*df_pts - off)/sv[0, 1]
    # ax.plot(df_pts, y_pts)
    return p_cw
        
        
