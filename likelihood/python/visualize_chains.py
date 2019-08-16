import h5py
# HDF5 groups are like python dictionaries, datasets are like numpy arrays
# https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/#basic-saving-and-reading-data

def read_chains(filename='../chains/LCDM_all_ell.h5')
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

def walker_plot(chain, params, extra_burnin_steps=0, theta_true=None, save_as_dir="../plots/", save_as_name="walkers.png"):
    #makes a walker plot and histogram
    #burnin_steps here means how many steps we discard when showing our plots. It doesn't have to match the burnin_steps argument to run
    #check theta_true!!
    nplots=len(params)
    #use gridspec??
    fig, axes = plt.subplots(nplots, 2, figsize=(10, 2.5*nplots))
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
        axes[i][1].plot(samples[:, :, i].T, color="k", alpha=0.4)   #should it be .T??
        if theta_true:
            axes[i][1].axhline(theta_true[i], color="#888888", lw=2)
        if i+1==nplots:
            axes[i][1].set_xlabel("steps")
        #axes[i][1].set_ylabel(p)

    fig.tight_layout(h_pad=0.0)
    fig.savefig(save_as_dir+save_as_name)
    plt.close()

    return 0

if __name__=='__main__':
    chain, logprob = read_chains(filename='../chains/LCDM_all_ell.h5')

    params=['h', 'omega_b', 'omega_cdm', 'tau_reio', 'A_s', 'n_s']
    walker_plot(chain, params)

    import IPython; IPython.embed()
