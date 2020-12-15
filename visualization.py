
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skc
import scipy.stats as sts

import general.plotting as gpl
import composite_tangling.code_analysis as ca


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
        
        
