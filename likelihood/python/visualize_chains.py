import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
import os
import h5py
import corner
# HDF5 groups are like python dictionaries, datasets are like numpy arrays
# https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/#basic-saving-and-reading-data

def read_chains(filename='../chains/LCDM_all_ell.h5'):
    with h5py.File(filename, 'r') as f:
        print(list(f.keys()))
        print(list(f['mcmc'].keys()))
        accepted = f['mcmc']['accepted'][:]
        chain = f['mcmc']['chain'][:]
        log_prob = f['mcmc']['log_prob'][:]

    return chain, log_prob

def get_median_and_errors(self):
    ps=np.percentile(self._sampler.flatchain, [16, 50, 84],axis=0)
    median=ps[1]
    err_plus=ps[2]-ps[1]
    err_minus=ps[1]-ps[0]
    return median, err_plus, err_minus

def walker_plot(samples, params, extra_burnin=0, theta_true=None, save_dir="../plots/", save_as_name="walkers.png"):
    #makes a walker plot and histogram
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #burnin_steps here means how many steps we discard when showing our plots. It doesn't have to match the burnin_steps argument to run
    samples=samples[extra_burnin:,:,:]
    #check theta_true!!
    nplots=len(params)
    #use gridspec??
    fig, axes = plt.subplots(nplots, 2, figsize=(8, 1.5*nplots))
    fig.subplots_adjust(wspace=0)
    if nplots==1:
        axes=[axes] #want 2D array to index below

    for i, p in enumerate(params):
        axes[i][0].hist(np.reshape(samples, (samples.shape[0]*samples.shape[1], samples.shape[2]))[:, i].T, bins=30, orientation='horizontal', alpha=.5)
        axes[i][0].yaxis.set_major_locator(MaxNLocator(5))
        axes[i][0].minorticks_off()
        axes[i][0].invert_xaxis()
        if theta_true:
            axes[i][0].axhline(theta_true[i], color="#888888", lw=2)
        axes[i][0].set_ylabel(p)
        if i+1==nplots:
            axes[i][0].set_xlabel("counts")

        axes[i][0].get_shared_y_axes().join(axes[i][0], axes[i][1])
        axes[i][1].minorticks_off()
        plt.setp(axes[i][1].get_yticklabels(), visible=False)
        # .T transposes the chains to plot each walker's position as a function of time
        axes[i][1].plot(samples[:, :, i], color="k", alpha=0.4)   #should it be .T??
        if theta_true:
            axes[i][1].axhline(theta_true[i], color="#888888", lw=2)
        if i+1==nplots:
            axes[i][1].set_xlabel("steps")
        #axes[i][1].set_ylabel(p)

    fig.tight_layout(h_pad=0.0)
    fig.savefig(save_dir+save_as_name)
    plt.show()
    plt.close()

    return 0

def triangle_plot(samples, params, extra_burnin=0, theta_true=None, save_dir="../plots/", save_as_name="corner.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    samples=samples[extra_burnin:,:,:]
    samples=np.reshape(samples, (samples.shape[0]*samples.shape[1], samples.shape[2]))
    fig = corner.corner(samples, labels=params, truths=theta_true)
    fig.savefig(save_dir+save_as_name)
    plt.show()
    plt.close()


def plot_smoothed_density(samples, label=None, n=51, ax=None, color=None, linestyle='-', linewidth=1, kde_fac=1, weights=None):
    density = stats.gaussian_kde(samples, bw_method='scott')
    density.set_bandwidth(bw_method=density.factor*kde_fac)

    x=np.linspace(np.amin(samples), np.amax(samples),n)
    bin_centers = 0.5*(x[1:]+x[:-1])

    if ax:
        ax.set_ylim(0, 1.1)
        ax.plot(bin_centers, density(bin_centers)/np.amax(density(bin_centers)), label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    else:
        plt.ylim(0, 1.1)
        plt.plot(bin_centers, density(bin_centers)/np.amax(density(bin_centers)), label=label, color=color, linestyle=linestyle, linewidth=linewidth)

def parameter_plot(chain_list, params, extra_burnin=0, theta_true=None, save_dir="../plots/", save_as_name="parameters.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ndim=len(params)
    rows=2
    colwidth=5
    cols=int(np.ceil(ndim/rows))
    fig, ax=plt.subplots(rows, cols, figsize=(colwidth*cols,colwidth/1.2*rows))
    for i, p in enumerate(params):
        ax_i=int(i/cols)
        ax_j=int(i%cols)

        for j, chain in enumerate(chain_list):
            chain=chain[extra_burnin:,:,:]
            flatchain=np.reshape(chain, (chain.shape[0]*chain.shape[1], chain.shape[2]))
            samples=flatchain[:,i]
            plot_smoothed_density(samples, ax=ax[ax_i,ax_j], kde_fac=2)#label=legend_list[j],
                    #, color=colors[j], linestyle=linestyles[j], linewidth=linewidths[j])
        ax[ax_i,ax_j].set_xlabel(p, fontsize=15)
        #ax[ax_i,ax_j].set_xlim(xlim_dict[param][0], xlim_dict[param][1])
    #plt.legend(loc='lower right', fontsize='x-large')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_dir+save_as_name, dpi=100)
    plt.show()
    plt.close()


if __name__=='__main__':
    chain, logprob = read_chains(filename='../chains/LCDM_all_ell.h5')
    #params=['h', 'omega_b', 'omega_cdm', 'tau_reio', 'A_s', 'n_s']
    params=[r'$h$', r'$\Omega_b$', r'$\Omega_{cdm}$', r'$\tau$', r'$A_s$', r'$n_s$']
    walker_plot(chain, params, extra_burnin=0)
    triangle_plot(chain, params, extra_burnin=0)
    parameter_plot([chain], params, extra_burnin=0)
