import numpy as np
from classy import Class

def loglike(cl_theory, compression_vectors, compressed_data):
    #compress cl and compute gaussian chi square
    chi_sq=0
    for p in compression_vectors:
        y_p_theory=compression_vectors[p].dot(cl_theory)
        y_p=compressed_data[p]
        chi_sq+=(y_p-y_p_theory)**2
    #return gaussian likelihood
    return -0.5*chi_sq

def logprior(values, names, prior_bound_dict, prior_gaussian_dict):
    '''
    places uniform priors from datafile
    places Gaussian priors from datafile

    values is an array the length of params (things that are varying)
    names is a list of parameter names in the same order as values
    prior_bound_dict is a dictionary of lists, the keys are the names and the lists are the lower and upper bounds for uniform priors
    prior_gaussian_dict is a dictionary of lists, the keys are the names and the lists are the mean and width for Gaussian priors
    '''
    for i, n in enumerate(names):
        val = values[i]
        low_bound = prior_bound_dict[n][0]
        up_bound = prior_bound_dict[n][1]
        assert low_bound < up_bound
        if not low_bound <= val <= up_bound:
            return -np.inf

    # this part will only run if all parameters are within the ranges allowed by the priors file
    log_prior_val=0
    for name in prior_gaussian_dict:
        val = values[names.index(name)]
        mu = prior_gaussian_dict[name][0]
        sigma = prior_gaussian_dict[name][1]
        log_prior_val += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(val-mu)**2/sigma**2

    return log_prior_val


def logprob(values, names, prior_bound_dict, prior_gaussian_dict, class_param_dict,  compression_vectors, compressed_data):
    '''
    values - an array the length of names (things that are varying)
    names - a list of parameter names in the same order as theta_p
    prior_bound_dict - a dictionary with upper and lower bounds for flat priors
    prior_gaussian_dict - a dictionary with mu and sigma for Gaussian priors
    model - a function that takes a parameter dictionary and returns the power spectrum
    class_param_dict - a dictionary with all of the necessary CLASS inputs apart from those given in names and values
    compression_vectors - a dictionary of compression vectors
    compressed_data - a dictionary of compressed data
    '''
    #priors first to make sure tau is in a sensible range and avoid errors in CLASS
    lp = logprior(values, names, prior_bound_dict, prior_gaussian_dict)
    if not np.isfinite(lp):
        return -np.inf

    #get cl from CLASS
    thetas=class_param_dict.copy()
    for i, n in enumerate(names):
        thetas[n]=values[i]
    #print('dictionary for CLASS', thetas)
    cl_theory=get_cl_theory(thetas)

    ll = loglike(cl_theory, compression_vectors, compressed_data)

    #combine likelihood and priors
    return lp + ll

def get_cl_theory(thetas, ellmin=2, ellmax=2508, T_cmb=2.7255):
    class_obj = Class()
    class_obj.set(thetas)
    class_obj.compute()
    cls = class_obj.lensed_cl(3000)
    class_obj.struct_cleanup()
    class_obj.empty()

    fac=(T_cmb*1e6)**2
    return (fac*cls['tt'])[ellmin:ellmax+1]
