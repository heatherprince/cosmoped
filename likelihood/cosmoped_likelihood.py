import numpy as np

def main():
    path='../compression_vectors/output/testing_5percent/' #get from inifile
    #path='../compression_vectors/output/from_old_code/'
    # cosmoped=CosMOPED(path, year=2015, spectra='TT', use_low_ell_TT=False)
    # cosmoped.test()
    cosmoped=CosMOPED(path, year=2015, spectra='TT', use_low_ell_TT=False)
    cosmoped.test()

    cosmoped=CosMOPED(path, year=2015, spectra='TT', use_low_ell_TT=True)
    cosmoped.test()

    cosmoped=CosMOPED(path, year=2015, spectra='TTTEEE', use_low_ell_TT=True)
    cosmoped.test()


class CosMOPED():
    def __init__(self, path, year=2015, spectra='TT', use_low_ell_TT=False):
        '''
        year = 2015 or 2018
        spectra = TT or TTTEEE
        use_low_ell_bins = True or False (refers to low-ell temperature bins)
        '''
        self.year=year
        self.spectra=spectra
        self.use_low_ell_TT=use_low_ell_TT

        # read in compression vectors and compressed data
        self.compression_vector_dict, self.compressed_data_dict = self.read_compression_vectors_and_compressed_data(path)

        # for computing theory vector [TT, TE, EE]
        # TT l=2-2508 (2-29 only used if requested)
        self.lmin = 2
        self.lmaxtt = 2508
        self.lmaxteee = 1996
        # TE and EE l=2-1996 (2-29 currently not used)

    def read_compression_vectors_and_compressed_data(self, path):
        if self.spectra=='TT' and not self.use_low_ell_TT:
            name_extra=str(self.year)+'_TT_high_ell'
        elif self.spectra=='TT' and self.use_low_ell_TT:
            name_extra=str(self.year)+'_TT_all_ell'
        elif self.spectra=='TTTEEE' and not self.use_low_ell_TT:
            name_extra=str(self.year)+'_TTTEEE_high_ell'
        elif self.spectra=='TTTEEE' and self.use_low_ell_TT:
            name_extra=str(self.year)+'_TT_all_TEEE_high_ell'

        f=open(path+'compressed_planck_data_'+name_extra+'.dat', 'r+')
        compressed_data_arr = [line.strip().split(' ') for line in f.readlines()
                    if not (line.startswith('#') or line.startswith('\n'))]
        f.close()
        param_names=np.array([row[0] for row in compressed_data_arr])
        param_compressed_values=np.array([float(row[1]) for row in compressed_data_arr])

        compressed_data_dict={}
        for i, p in enumerate(param_names):
            compressed_data_dict[p]=param_compressed_values[i]

        compression_vector_dict={}
        for p in param_names:
            compression_vector_dict[p]=np.loadtxt(path+p+'_compression_and_binning_vector_'+name_extra+'.dat')
        return compression_vector_dict, compressed_data_dict


    def loglike(self, Dltt, Dlte, Dlee, ellmin=2):
        # make cl_theory vector from Dltt, Dlte, Dlee
        ls=np.arange(len(Dltt))+ellmin
        fac=ls*(ls+1)/(2*np.pi)
        Cltt=Dltt/fac
        Clte=Dlte/fac
        Clee=Dlee/fac

        i_min = self.lmin-ellmin
        i_maxtt = self.lmaxtt+1-ellmin
        i_maxteee = self.lmaxteee+1-ellmin

        if self.spectra=='TT':
            cl_theory = Cltt[i_min:i_maxtt]
        else:
            cl_theory = np.concatenate((Cltt[i_min:i_maxtt], Clte[i_min:i_maxteee], Clee[i_min:i_maxteee]))


        #compress cl and compute gaussian chi square
        chi_sq=0
        for p in self.compression_vector_dict:
            y_p_theory=self.compression_vector_dict[p].dot(cl_theory)
            y_p=self.compressed_data_dict[p]
            chi_sq+=(y_p-y_p_theory)**2
        #return log likelihood
        return -0.5*chi_sq


    def test(self):
        ls, Dltt, Dlte, Dlee = np.genfromtxt('data/Dl_LCDM2015.dat', unpack=True) #
        ellmin=int(ls[0])
        print('Year='+str(self.year)+', spectra='+self.spectra+', use low ell TT='+str(self.use_low_ell_TT))
        print('cosmoped log-likelihood:',self.loglike(Dltt, Dlte, Dlee, ellmin))



if __name__=='__main__':
    main()