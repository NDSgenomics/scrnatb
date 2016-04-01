import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats

from GPclust import OMGP
from GPy.util.linalg import pdinv, dpotrs

import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_2d_gplvm_fit(gplvm):
    ''' Given a gplvm, predict 1-d latent variable fit in to 2-d data space
    '''
    X = gplvm.X.mean[:, [0]]
    tt = np.linspace(X.min(), X.max())[:, None]
    ttxy = gplvm.predict(tt)[0]
    plt.plot(*ttxy.T, c='w', lw=5)
    plt.plot(*ttxy.T, c='b')

class point_sprayer(object):
    ''' Class for quickly drawing point cloud examples. Useful for demonstrations.

    Only works with interactive backends. In Jupyter, do %matplotlib notebook.
    '''
    def __init__(self, ax, pix_err=1, std=0.1):
        self.canvas = ax.get_figure().canvas

        self.rv = stats.multivariate_normal(mean=[0, 0], cov=std ** 2)

        self.pt_lst = []
        self.pt_plot = ax.plot([], [], marker='o',
                               linestyle='none', zorder=5)[0]
        self.pix_err = pix_err

        self.connect()
        self.press = False

    def connect(self):
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)


    def on_press(self, event):
        self.press = event.xdata, event.ydata

    def on_motion(self, event):
        if self.press is None:
            return

        if event.button == 1:
            s = self.rv.rvs(1)
            self.pt_lst.append((event.xdata + s[0], event.ydata + s[1]))

    def on_release(self, event):
        self.press = None
        self.redraw()

    def redraw(self):
        if len(self.pt_lst) > 0:
            x, y = zip(*self.pt_lst)
        else:
            x, y = [], []

        self.pt_plot.set_xdata(x)
        self.pt_plot.set_ydata(y)
        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points in the format the rest of the
        code expects'''
        return np.vstack(self.pt_lst).T

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)



def predict_grid(bgplvm, resolution=50, which_indices=(0,1)):
    X = bgplvm.X.mean[:, which_indices]
    xi, yi = np.meshgrid(1.1 * np.linspace(X[:, 0].min(), X[:, 0].max(), resolution),
                         1.1 * np.linspace(X[:, 1].min(), X[:, 1].max(), resolution))

    pred_X = np.zeros((bgplvm.X.shape[1], resolution ** 2))
    pred_X[which_indices[0]] = xi.reshape(resolution ** 2)
    pred_X[which_indices[1]] = yi.reshape(resolution ** 2)

    pred_Y, pred_Y_var = bgplvm.predict(pred_X.T)

    extent = [xi.min(), xi.max(), yi.min(), yi.max()]

    return pred_Y, pred_Y_var, extent


def breakpoint_linear(x, ts, k1, k2, c1):
    '''Function representing a step-wise linear curve with one
    breakpoint located at ts.
    '''
    return np.piecewise(x, [x < ts], [lambda x: k1 * x + c1,
                                      lambda x: k2 * x + (k1 - k2) * ts + c1])


def identify_bifurcation_point(omgp, n_splits=30):
    mix_m = OMGP(omgp.X, omgp.Y, K=omgp.K, kernels=omgp.kern)
    mix_m.variance = omgp.variance

    phi = omgp.phi

    log_liks = []

    t_splits = np.linspace(mix_m.X.min(), mix_m.X.max(), n_splits)
    for t_s in tqdm(t_splits):
        mask = mix_m.X > t_s
        phi_mix = np.ones_like(phi) * 0.5
        phi_mix[mask[:, 0]] = phi[mask[:, 0]]

        mix_m.phi = phi_mix
        log_liks.append(mix_m.log_likelihood())

    x = t_splits
    y = np.array(log_liks)
    p, e = optimize.curve_fit(breakpoint_linear, x, y)

    return p[0]


def phase_trajectory(lat_pse_tim_r, known_time):
    t = lat_pse_tim_r
    t_pos = t - t.min()
    d = known_time

    @np.vectorize
    def align_objective(t0):
        return -stats.pearsonr((t_pos + t0) % t_pos.max(), d)[0] ** 2

    # One could use scipy.optimize to find the optimal phase. But in practice it is to
    # quick to evaluate the function that simple argmin on a grid works very well.

    xx = np.linspace(t_pos.min(), t_pos.max(), 200)
    yy = align_objective(xx)

    res = {'x': xx[yy.argmin()]}

    new_t = (t_pos + res['x']) % t_pos.max() + t.min()

    return new_t


def omgp_model_bound(omgp):
    ''' Calculate the part of the omgp bound which does not depend
    on the response variable.
    '''
    GP_bound = 0.0

    LBs = []
    # Precalculate the bound minus data fit,
    # and LB matrices used for data fit term.
    for i, kern in enumerate(omgp.kern):
        K = kern.K(omgp.X)
        B_inv = np.diag(1. / ((omgp.phi[:, i] + 1e-6) / omgp.variance))
        Bi, LB, LBi, Blogdet = pdinv(K + B_inv)
        LBs.append(LB)

        # Penalty
        GP_bound -= 0.5 * Blogdet

        # Constant
        GP_bound -= 0.5 * omgp.D * np.einsum('j,j->', omgp.phi[:, i], np.log(2 * np.pi * omgp.variance))

    model_bound = GP_bound + omgp.mixing_prop_bound() + omgp.H

    return model_bound, LBs


def bifurcation_statistics(omgp_gene, expression_matrix):
    ''' Given an OMGP model and an expression matrix, evaluate how well
    every gene fits the model.
    '''
    bif_stats = pd.DataFrame(index=expression_matrix.index)
    bif_stats['bif_ll'] = np.nan
    bif_stats['amb_ll'] = np.nan
    bif_stats['shuff_bif_ll'] = np.nan
    bif_stats['shuff_amb_ll'] = np.nan

    # Make a "copy" of provided OMGP but assign ambiguous mixture parameters
    omgp_gene_a = OMGP(omgp_gene.X, omgp_gene.Y,
                       K=omgp_gene.K,
                       kernels=[k.copy() for k in omgp_gene.kern],
                       prior_Z=omgp_gene.prior_Z,
                       variance=float(omgp_gene.variance))

    omgp_gene_a.phi = np.ones_like(omgp_gene.phi) * 1. / omgp_gene.K

    # To control FDR, perform the same likelihood calculation, but with permuted X values

    shuff_X = np.array(omgp_gene.X).copy()
    np.random.shuffle(shuff_X)

    omgp_gene_shuff = OMGP(shuff_X, omgp_gene.Y,
                           K=omgp_gene.K,
                           kernels=[k.copy() for k in omgp_gene.kern],
                           prior_Z=omgp_gene.prior_Z,
                           variance=float(omgp_gene.variance))

    omgp_gene_shuff_a = OMGP(shuff_X, omgp_gene.Y,
                             K=omgp_gene.K,
                             kernels=[k.copy() for k in omgp_gene.kern],
                             prior_Z=omgp_gene.prior_Z,
                             variance=float(omgp_gene.variance))

    omgp_gene_shuff_a.phi = np.ones_like(omgp_gene.phi) * 1. / omgp_gene.K

    # Precalculate response-variable independent parts
    omgps = [omgp_gene, omgp_gene_a, omgp_gene_shuff, omgp_gene_shuff_a]
    column_list = ['bif_ll', 'amb_ll', 'shuff_bif_ll', 'shuff_amb_ll']
    precalcs = [omgp_model_bound(omgp) for omgp in omgps]

    # Calculate the likelihoods of the models for every gene
    for gene in tqdm(expression_matrix.index):
        Y = expression_matrix.ix[gene]
        YYT = np.outer(Y, Y)

        for precalc, column in zip(precalcs, column_list):
            model_bound, LBs = precalc
            GP_data_fit = 0.
            for LB in LBs:
                GP_data_fit -= .5 * dpotrs(LB, YYT)[0].trace()

            bif_stats.ix[gene, column] = model_bound + GP_data_fit

    bif_stats['phi0_corr'] = expression_matrix.corrwith(pd.Series(omgp_gene.phi[:, 0], index=expression_matrix.columns), 1)
    bif_stats['D'] = bif_stats['bif_ll'] - bif_stats['amb_ll']
    bif_stats['shuff_D'] = bif_stats['shuff_bif_ll'] - bif_stats['shuff_amb_ll']

    return bif_stats
