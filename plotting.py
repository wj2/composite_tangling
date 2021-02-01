
import numpy as np

import matplotlib.pyplot as plt
import general.plotting as gpl

def plot_metrics(x_vals, metrics, x_label, y_labels=None, axs=None, fwid=3,
                 theory=None, eps=.1,  **kwargs):
    if theory is None:
        theory = {}
    if axs is None:
        fsize = (fwid*len(metrics), fwid)
        f, axs = plt.subplots(1, len(metrics), figsize=fsize)
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
            gpl.plot_trace_werr(x_vals, v_theor, linestyle='dashed', ax=ax)
        ax.set_title(k)
        ax.set_xlabel(x_label)
        if y_labels is not None:
            ax.set_ylabel(y_labels[k])
        yl = ax.get_ylim()
        if np.diff(yl) < eps:
            ax.set_ylim(yl[0] - eps, yl[1] + eps)
    return axs
