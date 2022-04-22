import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import scipy.linalg

import os
import sys
import shutil   #to copy files

import read_inifile
from classy import Class



def main():
    try:
        settings_inifile=sys.argv[1]
    except:
        print('ERROR: You need to provide an inifile as a command line argument e.g.')
        print('$ python compression_vectors.py path/to/inifile.ini')
        sys.exit(1)

    #read from inifile
    year, compression_inifile, data_dir, save_dir = read_inifile.read_settings_file(settings_inifile)

    compression_obj = CosmopedVectors(compression_inifile, data_dir, save_dir, year)
    derivatives = compression_obj.get_C_derivative()
    compression_obj.create_and_save_all_compression_vectors()
    compression_obj.compress_and_save_planck_data_all_combos()


class CosmopedVectors:
    def __init__(self, compression_inifile, data_dir, save_dir, year=2018):
        self.save_dir=save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        shutil.copy2(compression_inifile, self.save_dir)

        self.year=year
        self.param_dict, self.ordered_param_names=read_inifile.read_compression_file(compression_inifile)

        self.lmin_class = 2
        self.plmin_tt=2
        self.plmin=30
        self.plmax=2508
        self.calPlanck=1

        self.nbintt_lo = 2 # l=2-29
        self.nbintt_hi = 215 # l=30-2508
        self.nbinte = 199 # l=30-1996
        self.nbinee = 199 # l=30-1996

        self.nbin_hi=self.nbintt_hi+self.nbinte+self.nbinee
        self.nbintt=self.nbintt_hi+self.nbintt_lo
        self.nbin_tot=self.nbintt+self.nbinte+self.nbinee

        if year==2015:
            version=18
        elif year==2018:
            version=22
        self.data_dir=data_dir+'/planck'+str(year)+'_plik_lite/'
        self.data_dir_low_ell=data_dir+'/planck'+str(year)+'_low_ell/'

        self.cmb_file=self.data_dir+'cl_cmb_plik_v'+str(version)+'.dat'
        self.cov_file=self.data_dir+'c_matrix_plik_v'+str(version)+'.dat'
        self.blmin_file = self.data_dir+'blmin.dat'
        self.blmax_file = self.data_dir+'blmax.dat'
        self.binw_file = self.data_dir+'bweight.dat'

        self.cmb_file_low_ell=self.data_dir_low_ell+'CTT_bin_low_ell_'+str(year)+'.dat'

        # read in binned ell value, C(l) TT, TE and EE and errors
        self.b_lo, self.X_lo, self.X_error_lo=np.genfromtxt(self.cmb_file_low_ell, unpack=True)
        b_hi, X_hi, X_error_hi=np.genfromtxt(self.cmb_file, unpack=True)

        self.b = np.concatenate((self.b_lo, b_hi))
        self.X_data = np.concatenate((self.X_lo, X_hi))
        self.X_error = np.concatenate((self.X_error_lo, X_error_hi))

        # read in binning datafiles
        self.blmin=np.loadtxt(self.blmin_file).astype(int)
        self.blmax=np.loadtxt(self.blmax_file).astype(int)
        self.bin_w=np.loadtxt(self.binw_file)

        self.blmin_low_ell=np.loadtxt(self.data_dir_low_ell+'blmin_low_ell.dat').astype(int)
        self.blmax_low_ell=np.loadtxt(self.data_dir_low_ell+'blmax_low_ell.dat').astype(int)
        self.bin_w_low_ell=np.loadtxt(self.data_dir_low_ell+'bweight_low_ell.dat')

        # TT with low ell
        self.blmin_TT=np.concatenate((self.blmin_low_ell, self.blmin+len(self.bin_w_low_ell)))
        self.blmax_TT=np.concatenate((self.blmax_low_ell, self.blmax+len(self.bin_w_low_ell)))
        self.bin_w_TT=np.concatenate((self.bin_w_low_ell, self.bin_w))

        self.fisher_TT_lo = np.diag(1./self.X_error_lo**2)
        self.fisher_TT_hi = self.get_inverse_covmat(spectra='TT')
        self.fisher_TTTEEE_hi = self.get_inverse_covmat(spectra='TTTEEE')

        self.class_dict={
                'output': 'tCl,pCl,lCl',
                'l_max_scalars': 3000,
                'lensing': 'yes',
                'N_ur':2.03066666667, #1 massive neutrino to match camb
                'N_ncdm': 1,
                'omega_ncdm' : 0.0006451439,
                'YHe':0.245341,
                'non linear' : 'halofit'}
        self.class_dict.update(self.param_dict)

        self.T_cmb=2.7255

        #initilise later
        self.dCdtheta_dict_full=None

        self.compression_vectors_binned_data_TT_hi=None
        self.compression_vectors_binned_data_TT_all=None
        self.compression_vectors_binned_data_TTTEEE_hi=None
        self.compression_vectors_binned_data_TT_all_TEEE_hi=None


    def get_inverse_covmat(self, spectra):
        '''
        high ell TT, TE and EE
        from Planck plik-lite datafile
        '''
        f = FortranFile(self.cov_file, 'r')
        covmat = f.read_reals(dtype=float).reshape((self.nbin_hi,self.nbin_hi))
        f.close()

        for i in range(self.nbin_hi):
            for j in range(i,self.nbin_hi):
                covmat[i,j] = covmat[j,i]

        if spectra=='TT':
            #select relevant covmat for temperature only
            bin_no=self.nbintt_hi
            start=0
            end=start+bin_no
            cov=covmat[start:end, start:end]
        else: #TTTEEE
            bin_no=self.nbin_hi
            cov=covmat

        #invert high ell covariance matrix (cholesky decomposition should be faster)
        fisher=scipy.linalg.cho_solve(scipy.linalg.cho_factor(cov), np.identity(bin_no))
        fisher=fisher.transpose()

        return fisher


    def include_low_ell_in_fisher_matrix(fisher_lo, fisher_hi, spectra):
        bin_lo=self.nbintt_lo
        if spectra=='TT':
            bin_hi=self.nbintt_hi
        else:
            bin_hi=self.nbin_hi
        bin_no=bin_lo+bin_hi
        fisher=np.zeros(shape=(bin_no, bin_no))
        fisher[0:bin_lo, 0:bin_lo] = fisher_lo
        fisher[bin_lo:,bin_lo:] = fisher_hi

        return fisher


    def get_C_derivative(self):
        '''
        derivative of X=[C^TT_low_ell, C^TT_high_ell, C^TE_high_ell, C^EE_high_ell]
        with respect to different parameters for which we are computing compression vectors
        '''
        #use difference not fractional difference (for in case fiducial value is 0)
        params = self.ordered_param_names
        dtheta_diff = self.get_dtheta_diff()
        dCdtheta_dict_full = {}
        for param in params:
            print('computing numerical derivative of C with respect to '+param)
            theta_plus = self.class_dict.copy()
            theta_plus[param] += dtheta_diff[param]
            ell, C_plus = self.get_theoretical_TTTEEE_binned_power_spec(theta_plus, with_low_ell=True)

            theta_minus = self.class_dict.copy()
            theta_minus[param] -= dtheta_diff[param]
            ell, C_minus = self.get_theoretical_TTTEEE_binned_power_spec(theta_minus, with_low_ell=True)

            theta_2plus = self.class_dict.copy()
            theta_2plus[param] += 2*dtheta_diff[param]
            ell, C_2plus = self.get_theoretical_TTTEEE_binned_power_spec(theta_2plus, with_low_ell=True)

            theta_2minus = self.class_dict.copy()
            theta_2minus[param] -= 2*dtheta_diff[param]
            ell, C_2minus = self.get_theoretical_TTTEEE_binned_power_spec(theta_2minus, with_low_ell=True)

            # 5 point numerical derivative
            dCdtheta_dict_full[param]= (-C_2plus+8*C_plus-8*C_minus+C_2minus)/(12*dtheta_diff[param])
            #np.savetxt(self.save_dir+'/'+param+'_derivative_5pt.dat', dCdtheta_dict_full[param])

        self.dCdtheta_dict_full=dCdtheta_dict_full
        return dCdtheta_dict_full


    def get_dtheta_diff(self):
        #~10% of 1 sigma for numerical derivative
        theta_dict=self.param_dict
        if self.year==2015:
            errors_LCDM_planck={'h':0.0096, 'omega_b':0.00023, 'omega_cdm': 0.0022,
                    'tau_reio': 0.019, 'A_s': np.exp(0.036)/1e10, 'n_s': 0.0062,
                    'alpha_s': 0.016}   #dns/dlnk=n_run
        elif self.year==2018:
            errors_LCDM_planck={'h':0.005, 'omega_b':0.0001, 'omega_cdm': 0.001,
                    'tau_reio': 0.007, 'A_s': np.exp(0.016)/1e10, 'n_s': 0.004,
                    'alpha_s': 0.016}   #dns/dlnk=n_run
        dtheta_diff={}
        for p in self.ordered_param_names:
            try:
                print(p+': setting h=0.1 sigma for numerical derivative')
                dtheta_diff[p]=0.1*errors_LCDM_planck[p]
                # print('setting h=5% of parameter')
                # dtheta_diff[p]=0.05*theta_dict[p]
            except:
                print('I do not have the error for this parameter:', p, 'using default')
                if theta_dict[p]==0:
                    print(p+': setting h=0.01 for numerical derivative')
                    dtheta_diff[p]=0.01
                else:
                    print(p+': setting h=1% of fiducial parameter for numerical derivative')
                    dtheta_diff[p]=0.01*theta_dict[p]
        return dtheta_diff

    def get_theoretical_TT_TE_EE_unbinned_power_spec_D_ell(self, class_dict):
        ellmin = self.lmin_class
        ellmax = self.plmax
        cosmo = Class()
        cosmo.set(class_dict)
        cosmo.compute()
        cls = cosmo.lensed_cl(3000)
        cosmo.struct_cleanup()
        cosmo.empty()

        #get in units of microkelvin squared
        T_fac=(self.T_cmb*1e6)**2

        ell=cls['ell']
        D_fac=ell*(ell+1.)/(2*np.pi)

        Dltt=(T_fac*D_fac*cls['tt'])[ellmin:ellmax+1]
        Dlte=(T_fac*D_fac*cls['te'])[ellmin:ellmax+1]
        Dlee=(T_fac*D_fac*cls['ee'])[ellmin:ellmax+1]
        return cls['ell'][ellmin:ellmax+1], Dltt, Dlte, Dlee


    def get_theoretical_TTTEEE_binned_power_spec(self, class_dict, with_low_ell):
        # returns C_ell
        ell, Dltt, Dlte, Dlee = self.get_theoretical_TT_TE_EE_unbinned_power_spec_D_ell(class_dict)
        # bin D in the same way Planck 2015 TT ell>30 is binned
        ell_bin, X_model=self.bin_TTTEEE_power_spec_high_ell(Dltt, Dlte, Dlee)

        if with_low_ell:
            ell_bin_lo, Cltt_bin_lo, binned_D_fac_lo=self.bin_TT_power_spec_low_ell(Dltt)
            ell_bin=np.concatenate((ell_bin_lo, ell_bin))
            X_model=np.concatenate((Cltt_bin_lo, X_model))

        return ell_bin, X_model



    def bin_TTTEEE_power_spec_high_ell(self, Dltt, Dlte, Dlee):
        # return binned C(l) from numpy array of D(l) which starts at D(ellmin)
        ellmin = self.lmin_class
        blmin = self.blmin
        blmax = self.blmax
        bin_w = self.bin_w
        plmin = self.plmin


        ls=np.arange(Dltt.shape[0]+ellmin)[ellmin:]
        fac=ls*(ls+1)/(2*np.pi)
        Cltt=Dltt/fac
        Clte=Dlte/fac
        Clee=Dlee/fac

        Cltt_bin=np.zeros(self.nbintt_hi)
        for i in range(self.nbintt_hi):
            Cltt_bin[i]=np.sum(Cltt[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1]) #what happens in Fortran when blmin is 0? ok because plmin=30?

        Clte_bin=np.zeros(self.nbinte)
        for i in range(self.nbinte):
            Clte_bin[i]=np.sum(Clte[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])

        Clee_bin=np.zeros(self.nbinee)
        for i in range(self.nbinee):
            Clee_bin[i]=np.sum(Clee[blmin[i]+plmin-ellmin:blmax[i]+plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])


        X_model=np.zeros(self.nbin_hi)
        X_model[:self.nbintt_hi]=Cltt_bin/self.calPlanck**2
        X_model[self.nbintt_hi:self.nbintt_hi+self.nbinte]=Clte_bin/self.calPlanck**2
        X_model[self.nbintt_hi+self.nbinte:]=Clee_bin/self.calPlanck**2

        binned_ells=np.concatenate((self.plmin+0.5*(blmin[0:self.nbintt_hi]+blmax[0:self.nbintt_hi]), self.plmin+0.5*(blmin[0:self.nbinte]+blmax[0:self.nbinte]), self.plmin+0.5*(blmin[0:self.nbinee]+blmax[0:self.nbinee]) ))

        return binned_ells, X_model

    def bin_TT_power_spec_low_ell(self,Dltt):
        nbin_low_ell=2
        lmin_low_ell=[2, 16]
        lmax_low_ell=[15, 29]
        ellmin = self.lmin_class
        # return binned C(l) from numpy array of D(l) which starts at D(ellmin)
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

        Cltt_bin_lo=D_bin_lo/binned_D_fac/self.calPlanck**2

        return ell_bin_lo, Cltt_bin_lo, binned_D_fac



    def create_and_save_all_compression_vectors(self):
        # save compression vectors for unbinned data to use for data compression (in object)
        print('creating and saving compression vectors from derivatives')
        self.compression_vectors_binned_data_TT_hi=self.create_and_save_compression_vectors(spectra='TT', with_low_ell_TT=False, name_extra='TT_high_ell')
        self.compression_vectors_binned_data_TT_all=self.create_and_save_compression_vectors(spectra='TT', with_low_ell_TT=True, name_extra='TT_all_ell')
        self.compression_vectors_binned_data_TTTEEE_hi=self.create_and_save_compression_vectors(spectra='TTTEEE', with_low_ell_TT=False, name_extra='TTTEEE_high_ell')
        self.compression_vectors_binned_data_TT_all_TEEE_hi=self.create_and_save_compression_vectors(spectra='TTTEEE', with_low_ell_TT=True, name_extra='TT_all_TEEE_high_ell')

    def compress_and_save_planck_data_all_combos(self):
        print('compressing and saving Planck '+str(self.year)+' data')
        self.compress_and_save_planck_data(spectra='TT', with_low_ell_TT=False, name_extra='TT_high_ell')
        self.compress_and_save_planck_data(spectra='TT',with_low_ell_TT=True, name_extra='TT_all_ell')
        self.compress_and_save_planck_data(spectra='TTTEEE', with_low_ell_TT=False, name_extra='TTTEEE_high_ell')
        self.compress_and_save_planck_data(spectra='TTTEEE', with_low_ell_TT=True, name_extra='TT_all_TEEE_high_ell')


    def create_and_save_compression_vectors(self, spectra, with_low_ell_TT, name_extra):
        fisher=self.get_relevant_fisher_matrix(spectra, with_low_ell_TT)
        dCdtheta_dict = self.select_relevant_parts(self.dCdtheta_dict_full, spectra, with_low_ell_TT)

        compression_vecs_binned_data = self.get_compression_vectors_for_binned_data(dCdtheta_dict, fisher)
        compression_vecs_unbinned_data = self.get_compression_vectors_for_unbinned_data_from_binned(compression_vecs_binned_data, spectra, with_low_ell_TT)

        # save to file
        for p in compression_vecs_binned_data:
            np.savetxt(self.save_dir+'/'+p+'_compression_and_binning_vector_'+str(self.year)+'_'+name_extra+'.dat', compression_vecs_unbinned_data[p])
            np.savetxt(self.save_dir+'/'+p+'_compression_vector_'+str(self.year)+'_'+name_extra+'.dat', compression_vecs_binned_data[p])

        return compression_vecs_binned_data


    def compress_and_save_planck_data(self, spectra, with_low_ell_TT, name_extra):
        # select compression vector corresponding to spectra and with_low_ell_TT
        # choose correct bit of planck data vector
        if spectra=='TT' and not with_low_ell_TT:
            # TT l>=30
            compression_vecs=self.compression_vectors_binned_data_TT_hi
        elif spectra=='TT' and with_low_ell_TT:
            # TT all ell
            compression_vecs=self.compression_vectors_binned_data_TT_all
        elif spectra=='TTTEEE' and not with_low_ell_TT:
            # TT, TE, EE high ell
            compression_vecs=self.compression_vectors_binned_data_TTTEEE_hi
        elif spectra=='TTTEEE' and with_low_ell_TT:
            # TT all ell, TE and EE high ell
            compression_vecs=self.compression_vectors_binned_data_TT_all_TEEE_hi

        Cl_data = self.select_relevant_parts(self.X_data, spectra, with_low_ell_TT)
        # iterate over parameters
        # compress it
        # save in an appropriately named datafile
        # save to object
        compressed_planck_data={}
        f = open(self.save_dir+'/compressed_planck_data_'+str(self.year)+'_'+name_extra+'.dat', 'w')
        f.write('# compressed planck data \n')
        for p in compression_vecs:
            compressed_planck_data[p]=compression_vecs[p].dot(Cl_data)
            f.write(p+' '+str(compressed_planck_data[p])+'\n')
        f.close()
        return compressed_planck_data

    def select_relevant_parts(self, vector, spectra, with_low_ell_TT):
        if spectra=='TT' and not with_low_ell_TT:
            # TT l>=30
            i_min = self.nbintt_lo
            i_max = self.nbintt_lo + self.nbintt_hi
        elif spectra=='TT' and with_low_ell_TT:
            # TT all ell
            i_min = 0
            i_max = self.nbintt
        elif spectra=='TTTEEE' and not with_low_ell_TT:
            # TT, TE, EE high ell
            i_min = self.nbintt_lo
            i_max = self.nbintt_lo + self.nbin_hi
        elif spectra=='TTTEEE' and with_low_ell_TT:
            # TT all ell, TE and EE high ell
            i_min = 0
            i_max = self.nbin_tot

        if type(vector)==dict:
            # we have a dictionary of vectors: iterate through them
            new_dict = {}
            for p in vector:
                new_dict[p] = vector[p][i_min:i_max]
            return new_dict
        else:
            return vector[i_min:i_max]

    def get_relevant_fisher_matrix(self, spectra, with_low_ell_TT):
        if spectra=='TT' and not with_low_ell_TT:
            # TT l>=30
            fisher = self.fisher_TT_hi
        elif spectra=='TT' and with_low_ell_TT:
            # TT all ell
            bin_no=self.nbintt
            fisher=np.zeros(shape=(bin_no, bin_no))
            fisher[:self.nbintt_lo, :self.nbintt_lo]=self.fisher_TT_lo
            fisher[self.nbintt_lo:, self.nbintt_lo:] = self.fisher_TT_hi
        elif spectra=='TTTEEE' and not with_low_ell_TT:
            # TT, TE, EE high ell
            fisher = self.fisher_TTTEEE_hi
        elif spectra=='TTTEEE' and with_low_ell_TT:
            # TT all ell, TE and EE high ell
            bin_no=self.nbin_tot
            fisher=np.zeros(shape=(bin_no, bin_no))
            fisher[:self.nbintt_lo, :self.nbintt_lo]=self.fisher_TT_lo
            fisher[self.nbintt_lo:, self.nbintt_lo:] = self.fisher_TTTEEE_hi
        return fisher


    def get_compression_vectors_for_binned_data(self, dCdtheta_dict, fisher):
        compression_vecs={}
        params=list(dCdtheta_dict.keys())
        for i, param in enumerate(params):
            dCl=dCdtheta_dict[param]

            num=fisher.dot(dCl)
            denom_sq=dCl.dot(fisher.dot(dCl))

            if i==0:
                compression_vecs[param]=num/np.sqrt(denom_sq)

            else:
                num_sum=0
                denom_sum=0
                for q in range(i):
                    p_q=params[q]
                    num_sum+=dCl.dot(compression_vecs[p_q])*compression_vecs[p_q]
                    denom_sum+=(dCl.dot(compression_vecs[p_q]))**2

                compression_vecs[param]=(num-num_sum)/np.sqrt(denom_sq-denom_sum)

        return compression_vecs

    def get_compression_vectors_for_unbinned_data_from_binned(self, binned_vecs, spectra, with_low_ell_TT):
        '''
        returns compression vectors that both bin and compress theoretical CMB
        power spectra in one step
        '''
        unbinned_vecs={}
        for p in binned_vecs:
            unbinned_vecs[p]=self.include_binning_in_compression_vec(binned_vecs[p], spectra, with_low_ell_TT)
        return unbinned_vecs

    def include_binning_in_compression_vec(self, binned_compression_vec, spectra, with_low_ell_TT):
        # for ell>30 use Planck plik-lite binning weights
        blmin=self.blmin
        blmax=self.blmin
        bin_w=self.bin_w

        if with_low_ell_TT:
            compression_vec_low_ell=self.bin_w_low_ell.copy()
            for i in range(self.nbintt_lo):
                compression_vec_low_ell[self.blmin_low_ell[i]:self.blmax_low_ell[i]+1] *=binned_compression_vec[i] / self.calPlanck**2

            nbin_lo=self.nbintt_lo

        else:
            nbin_lo=0
            #low ell bit set to 0 so same length for all weighting vectors
            compression_vec_low_ell=np.zeros(self.plmin-self.plmin_tt)


        compression_vec_high_ell_TT=self.bin_w.copy()
        for i in range(self.nbintt_hi):
            compression_vec_high_ell_TT[self.blmin[i]:self.blmax[i]+1] *= binned_compression_vec[i+nbin_lo]/self.calPlanck**2
        compression_vec_high_ell_TT=compression_vec_high_ell_TT[0:self.blmax[i]+1]

        compression_vec=np.concatenate((compression_vec_low_ell, compression_vec_high_ell_TT))

        if spectra=='TTTEEE': #use TE and EE
            #import IPython; IPython.embed()
            # leave space to include EE compression later (and TE for consistency)
            compression_vec_low_ell_pol=np.zeros(self.plmin-self.plmin_tt)

            compression_vec_high_ell_TE=self.bin_w.copy()
            for i in range(self.nbinte):
                compression_vec_high_ell_TE[self.blmin[i]:self.blmax[i]+1] *= binned_compression_vec[i+self.nbintt_hi+nbin_lo]/self.calPlanck**2
            compression_vec_high_ell_TE=compression_vec_high_ell_TE[0:self.blmax[i]+1]

            compression_vec_high_ell_EE=self.bin_w.copy()
            for i in range(self.nbinee):
                compression_vec_high_ell_EE[self.blmin[i]:self.blmax[i]+1] *= binned_compression_vec[i+self.nbintt_hi+self.nbinte+nbin_lo]/self.calPlanck**2
            compression_vec_high_ell_EE=compression_vec_high_ell_EE[0:self.blmax[i]+1]

            compression_vec=np.concatenate((compression_vec, compression_vec_low_ell_pol, compression_vec_high_ell_TE, compression_vec_low_ell_pol, compression_vec_high_ell_EE))

        return compression_vec





if __name__=='__main__':
    main()
