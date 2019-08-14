import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import scipy.linalg
import copy

from classy import Class

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Binning constants
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plmin=30
plmax=2508
calPlanck=1

#currently just does TT - plan to extend later
nbintt = 215
nbinte = 199
nbinee = 199
nbin=nbintt+nbinte+nbinee

nbin_low_ell=2
lmin_low_ell=[2, 16]
lmax_low_ell=[15, 29]

low_ell_filename='../cmb_data/planck2015_low_ell/low_ell_bins.dat'  #update to new format
data_dir='../cmb_data/planck2015_plik_lite/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Binning
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def bin_TT_power_spec_high_ell(Dltt, ellmin=2):
    #return binned C(l)
    blmin=np.loadtxt(data_dir+'blmin.dat').astype(int)
    blmax=np.loadtxt(data_dir+'blmax.dat').astype(int)
    bin_w=np.loadtxt(data_dir+'bweight.dat')

    ls=np.arange(Dltt.shape[0]+ellmin)[ellmin:]
    fac=ls*(ls+1)/(2*np.pi)
    Cltt=Dltt/fac

    Cltt_bin=np.zeros(nbintt)

    for i in range(nbintt):
        Cltt_bin[i]=np.sum(Cltt[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1]) #what happens in Fortran when blmin is 0? ok because plmin=30?

    Cltt_bin/=calPlanck**2

    binned_ells=plmin+0.5*(blmin[0:nbintt]+blmax[0:nbintt])

    return binned_ells, Cltt_bin

def bin_TT_power_spec_low_ell(Dltt, ellmin=2):
    ls=np.arange(Dltt.shape[0]+ellmin)[ellmin:]
    fac=ls*(ls+1)/(2*np.pi)
    Cltt=Dltt/fac

    i_min_0=lmin_low_ell[0]-ellmin
    i_max_0=lmax_low_ell[0]+1-ellmin
    len0=i_max_0-i_min_0
    i_min_1=lmin_low_ell[1]-ellmin
    i_max_1=lmax_low_ell[1]+1-ellmin
    len1=i_max_1-i_min_1

    ell_0=np.arange(lmin_low_ell[0],lmax_low_ell[0]+1)
    ell_1=np.arange(lmin_low_ell[1],lmax_low_ell[1]+1)

    ell_bin_lo=np.array([np.mean(ell_0), np.mean(ell_1)])

    D_bin_0=np.mean(ell_0*(ell_0+1)/(2*np.pi)*Cltt[i_min_0:i_max_0])
    D_bin_1=np.mean(ell_1*(ell_1+1)/(2*np.pi)*Cltt[i_min_1:i_max_1])
    D_bin_lo=np.array([D_bin_0, D_bin_1])

    D_0_fac=np.sum(ell_0*(ell_0+1)/(2*np.pi))/len(ell_0)
    D_1_fac=np.sum(ell_1*(ell_1+1)/(2*np.pi))/len(ell_1)
    binned_D_fac=np.array([D_0_fac, D_1_fac])

    Cltt_bin_lo=D_bin_lo/binned_D_fac/calPlanck**2

    return ell_bin_lo, Cltt_bin_lo, binned_D_fac

def include_binning_in_compression_vec(binned_compression_vec, with_low_ell, ellmin=2):
    # for ell>30 use Planck plik-lite binning weights
    blmin=np.loadtxt(data_dir+'blmin.dat').astype(int)
    blmax=np.loadtxt(data_dir+'blmax.dat').astype(int)
    bin_w=np.loadtxt(data_dir+'bweight.dat')

    if with_low_ell:
        #need to include binning for two low ell bins: average in Dl = l(l+1) weighting in C
        ell_lo=np.arange(ellmin, plmin) #2-30
        D_fac=ell_lo*(ell_lo+1)
        compression_vec_low_ell=D_fac.astype(np.float64)

        ell_0=np.arange(lmin_low_ell[0],lmax_low_ell[0]+1)
        ell_1=np.arange(lmin_low_ell[1],lmax_low_ell[1]+1)
        D_0_fac=np.sum(ell_0*(ell_0+1))
        D_1_fac=np.sum(ell_1*(ell_1+1))
        norm_low_ell=[D_0_fac, D_1_fac]

        for i in range(nbin_low_ell):
            compression_vec_low_ell[lmin_low_ell[i]-ellmin:lmax_low_ell[i]+1-ellmin] *=binned_compression_vec[i] / calPlanck**2 / norm_low_ell[i]

        compression_vec_high_ell=bin_w.copy()
        for i in range(nbintt):
            compression_vec_high_ell[blmin[i]:blmax[i]+1] *= binned_compression_vec[i+nbin_low_ell]/calPlanck**2
        compression_vec_high_ell=compression_vec_high_ell[0:blmax[i]+1]


    else:
        #just high ell: new weighting vector will be bin_w *compression vector
        compression_vec_high_ell=bin_w.copy()
        for i in range(nbintt):
            compression_vec_high_ell[blmin[i]:blmax[i]+1] *= binned_compression_vec[i]/calPlanck**2
        compression_vec_high_ell=compression_vec_high_ell[0:blmax[i]+1]

        #low ell bit set to 0 so same length for all weighting vectors
        compression_vec_low_ell=np.zeros(plmin-ellmin)

    compression_vec=np.concatenate((compression_vec_low_ell, compression_vec_high_ell))
    return compression_vec

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model (theoretical power spectrum from CLASS)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_theoretical_TT_binned_power_spec(params, with_low_ell):
    ell, Dltt = get_theoretical_TT_unbinned_power_spec_D_ell(params)
    # bin D in the same way Planck 2015 TT ell>30 is binned
    ell_bin, Cltt_bin=bin_TT_power_spec_high_ell(Dltt)

    if with_low_ell:
        ell_bin_lo, Cltt_bin_lo, binned_D_fac_lo=bin_TT_power_spec_low_ell(Dltt)
        ell_bin=np.concatenate((ell_bin_lo, ell_bin))
        Cltt_bin=np.concatenate((Cltt_bin_lo, Cltt_bin))

    return ell_bin, Cltt_bin


def get_theoretical_TT_unbinned_power_spec_D_ell(class_params, ellmin=2, ellmax=plmax):
    cosmo = Class()
    cosmo.set(class_params)
    cosmo.compute()
    cls = cosmo.lensed_cl(3000)
    cosmo.struct_cleanup()
    cosmo.empty()

    #get in units of microkelvin squared
    T_cmb=2.7255
    fac=cls['ell']*(cls['ell']+1)/(2*np.pi)*(T_cmb*1e6)**2

    Dltt=(fac*cls['tt'])[ellmin:ellmax+1]
    return cls['ell'][ellmin:ellmax+1], Dltt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model derivatives (using theoretical binned power spectrum from CLASS)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_numerical_derivative_of_C_l_TT(theta_fid, param, with_low_ell, frac=0.01):
    '''
    theta_fid=dictionary of fiducial values
    param=index of parameter we are taking a derivative with respect to
    model=function giving theoretical spectrum
    '''

    theta_plus=theta_fid.copy()
    if theta_fid[param]==0:
        theta_plus[param]=frac
    else:
        theta_plus[param]=theta_plus[param]*(1+frac)
    ell, C_plus=get_theoretical_TT_binned_power_spec(theta_plus, with_low_ell)

    theta_minus=theta_fid.copy()
    if theta_fid[param]==0:
        theta_minus[param]=-frac
    else:
        theta_minus[param]=theta_minus[param]*(1-frac)
    ell, C_minus=get_theoretical_TT_binned_power_spec(theta_minus, with_low_ell)

    theta_diff=2*frac*theta_fid[param]

    dCdtheta=(C_plus-C_minus)/theta_diff

    return ell, dCdtheta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Data (Planck 2015)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_planck_TT_binned_power_spec(with_low_ell):
    #ell>30 binned ell, C, dC
    b, C, dC=get_planck_TT_binned_power_spec_high_ell()

    if with_low_ell:
        b_lo, C_lo, dC_lo=get_planck_TT_binned_power_spec_low_ell()
        b=np.concatenate((b_lo, b))
        C=np.concatenate((C_lo, C))
        dC=np.concatenate((dC_lo, dC))

    return b, C, dC

def get_planck_TT_binned_power_spec_high_ell():
    ell_bin, Cl, dCl=np.genfromtxt(data_dir+'cl_cmb_plik_v18.dat', unpack=True)
    return ell_bin[0:nbintt], Cl[0:nbintt], dCl[0:nbintt]

def get_planck_TT_binned_power_spec_low_ell():
    lo=np.loadtxt(low_ell_filename)
    b_lo=lo[:,0]
    D_lo=lo[:,1]
    err_D_lo=lo[:,2]

    #now need to convert D to C to match Planck datafile
    ell_0=np.arange(2, 16)
    ell_1=np.arange(16, 30)
    D_0_fac=np.sum(ell_0*(ell_0+1)/(2*np.pi))/len(ell_0)
    D_1_fac=np.sum(ell_1*(ell_1+1)/(2*np.pi))/len(ell_1)
    binned_D_fac=np.array([D_0_fac, D_1_fac])

    C_lo=D_lo/binned_D_fac
    dC_lo=err_D_lo/binned_D_fac

    return b_lo, C_lo, dC_lo

def get_inverse_covmat_TT(with_low_ell):
    #read full covmat
    f = FortranFile(data_dir+'c_matrix_plik_v18.dat', 'r')
    covmat = f.read_reals(dtype=float).reshape((nbin,nbin))
    f.close()

    for i in range(nbin):
        for j in range(i,nbin):
            covmat[i,j] = covmat[j,i]

    #select relevant covmat for temperature only
    bin_no=nbintt
    start=0
    end=start+bin_no
    cov=covmat[start:end, start:end]

    # if with_low_ell:
    #     bin_no=nbintt+2
    #     covmat_with_lo=np.zeros(shape=(bin_no, bin_no))
    #     b_lo, C_lo_TT, dC_lo_TT=get_planck_TT_binned_power_spec_low_ell()
    #     print('dC_lo_TT', dC_lo_TT)
    #     covmat_with_lo[0:2, 0:2]=np.diag(dC_lo_TT**2)
    #     covmat_with_lo[2:,2:]=cov
    #     cov=covmat_with_lo


    #invert high ell covariance matrix (cholesky decomposition should be faster)
    fisher=scipy.linalg.cho_solve(scipy.linalg.cho_factor(cov), np.identity(bin_no))
    #zack transposes it (because fortran indexing works differently?) but I don't think this should make a difference? check!
    fisher=fisher.transpose()

    if with_low_ell:
        bin_no=nbintt+2
        inv_covmat_with_lo=np.zeros(shape=(bin_no, bin_no))
        bin_no=nbintt+2
        b_lo, C_lo_TT, dC_lo_TT=get_planck_TT_binned_power_spec_low_ell()
        #print('dC_lo_TT', dC_lo_TT)
        inv_covmat_with_lo[0:2, 0:2]=np.diag(1./dC_lo_TT**2)
        inv_covmat_with_lo[2:,2:]= fisher
        fisher=inv_covmat_with_lo

    return fisher


if __name__=='__main__':
    #usage examples
    theta = values_LCDM_planck2015 = {'H0':67.31, 'omegabh2':0.02222,
                                    'omegach2': 0.1197, 'tau': 0.078,
                                    'As': np.exp(3.089)/1e10, 'ns': 0.9655}
    params = {
            'output': 'tCl,pCl,lCl',
            'l_max_scalars': 3000,
            'lensing': 'yes',
            'A_s': theta['As'],
            'n_s': theta['ns'],
            'h': theta['H0']/100.,
            'omega_b': theta['omegabh2'],
            'omega_cdm': theta['omegach2'],
            'tau_reio': theta['tau'],
            'N_ur':2.046, #1 massive neutrino to match camb
            'N_ncdm': 1,
            'm_ncdm' : 0.06}

    # get theoretical temperature power spectrum D using CLASS
    ell, Dltt = get_theoretical_TT_unbinned_power_spec_D_ell(params)
    # bin D in the same way Planck 2015 TT ell>30 is binned
    ell_bin, Cltt_bin=bin_TT_power_spec_high_ell(Dltt)
    # bin D in two low ell bins then convert to C sensibly
    ell_bin_lo, Cltt_bin_lo, binned_D_fac_lo=bin_TT_power_spec_low_ell(Dltt)

    #read Planck 2015 ell>30 binned power spectrum
    ell_bin_planck, Cltt_bin_planck, dCltt_bin_planck=get_planck_TT_binned_power_spec_high_ell()
    D_fac_planck=ell_bin_planck*(ell_bin_planck+1)/(2*np.pi)

    #read Planck 2015 ell<30 binned power spectrum
    b_lo_planck, C_lo_planck, dC_lo_planck=get_planck_TT_binned_power_spec_low_ell()

    plt.plot(ell, Dltt, color='black', label='unbinned theory')
    plt.scatter(ell_bin_lo, binned_D_fac_lo*Cltt_bin_lo, s=15, label='binned low ell theory', c='green')
    plt.errorbar(b_lo_planck, binned_D_fac_lo*C_lo_planck, yerr=binned_D_fac_lo*dC_lo_planck, linestyle='None', label='binned low ell planck2015')
    plt.scatter(ell_bin, ell_bin*(ell_bin+1)/(2*np.pi)*Cltt_bin, s=15, label='binned high ell theory', c='red')
    plt.errorbar(ell_bin_planck, D_fac_planck*Cltt_bin_planck, yerr=D_fac_planck*dCltt_bin_planck, linestyle='None', label='binned high ell planck2015')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # # get Planck 2015 all ell binned power spectrum
    # b_planck, C_planck, dC_planck=get_planck_TT_binned_power_spec(with_low_ell=True)
    # D_fac_all_ell=b_planck*(b_planck+1)/(2*np.pi)
    # # get theoretical binned power spectrum
    # ell_bin, Cltt_bin=get_theoretical_TT_binned_power_spec(params, with_low_ell=True)
    # # plot
    # plt.errorbar(b_planck, D_fac_all_ell*C_planck, yerr=D_fac_all_ell*dC_planck, linestyle='none', marker='+', label='binned planck2015')
    # plt.scatter(ell_bin, ell_bin*(ell_bin+1)/(2*np.pi)*Cltt_bin, s=30, c='black', marker='x', label='binned theory')
    # plt.legend()
    # plt.xscale('log')
    # plt.show()
    #
    # fisher_high_ell=get_inverse_covmat_TT(with_low_ell=False)
    # fisher=get_inverse_covmat_TT(with_low_ell=True)
    # plt.pcolormesh(fisher)
    # plt.colorbar()
    # plt.show()
    #
    # plt.pcolormesh(fisher_high_ell-fisher[2:,2:])
    # plt.colorbar()
    # plt.show()
    #
    # param='h'
    # ell, dCdtheta=get_numerical_derivative_of_C_l_TT(params, param, with_low_ell=True, frac=0.01)
    # plt.plot(ell, ell*(ell+1)/(2*np.pi)*dCdtheta, label='derivative wrt h')
    # plt.show()
