import numpy as np
import matplotlib.pyplot as plt
import time

import os
import shutil   #to copy files

import cmb_angular_power_spectrum
import read_ini_file

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# main function that creates compression vectors and compresses the planck data
# and saves them along with the ini file used to create them
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_and_save_compression_data(ini_filename='../inifiles/LambdaCDM.ini', with_low_ell=True):
    #read ini file
    param_dict, ordered_param_names=read_ini_file.read_file(ini_filename)
    #create compression vectors for binned planck data
    compression_vecs=get_compression_vectors_for_binned_data(param_dict, ordered_param_names, with_low_ell)
    #compress planck data
    compressed_planck_data=compress_planck_binned_TT(compression_vecs, with_low_ell)
    #create vectors that compress and bin theoretical power spectrum in one step
    compression_and_binning_vecs=get_compression_vectors_for_unbinned_data_from_binned(compression_vecs, with_low_ell)

    #save in sensible folder with .ini file
    folder_name=os.path.splitext(os.path.basename(ini_filename))[0]

    path='../output/'+folder_name+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    shutil.copy2(ini_filename, path)

    if with_low_ell:
        filename_extra='with_low_ell'
    else:
        filename_extra='high_ell_only'

    #save each compression vector
    for p in compression_and_binning_vecs:
        np.savetxt(path+p+'_compression_vector_'+filename_extra+'.dat', compression_and_binning_vecs[p])

    #save compressed data in one file with param name and compressed value
    f = open(path+'compressed_planck_data_'+filename_extra+'.dat', 'w')
    f.write('# compressed planck data \n')
    for p in ordered_param_names:
        f.write(p+' '+str(compressed_planck_data[p])+'\n')
    f.close()

    #get ell range
    class_params=get_class_dict(param_dict)
    ell, _=cmb_angular_power_spectrum.get_theoretical_TT_unbinned_power_spec_D_ell(class_params)
    np.savetxt(path+'ell.dat', ell)

    return compression_and_binning_vecs, compressed_planck_data


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compress the data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compress_planck_binned_TT(compression_vectors, with_low_ell):
    ell_bin, Cltt_bin, dCltt_bin=cmb_angular_power_spectrum.get_planck_TT_binned_power_spec(with_low_ell)

    compressed_data_dict={}
    for p in compression_vectors:
        compressed_data_dict[p]=compression_vectors[p].dot(Cltt_bin)

    return compressed_data_dict

def compress_theory_binned_TT(compression_vectors, theta_dict, with_low_ell):
    class_params=get_class_dict(theta_dict)
    ell_bin, Cltt_bin=cmb_angular_power_spectrum.get_theoretical_TT_binned_power_spec(class_params, with_low_ell)

    compressed_data_dict={}
    for p in compression_vectors:
        compressed_data_dict[p]=compression_vectors[p].dot(Cltt_bin)

    return compressed_data_dict

def compress_and_bin_theory_TT(compression_and_binning_vectors, theta_dict):
    class_params=get_class_dict(theta_dict)
    ell, Dltt_theory=cmb_angular_power_spectrum.get_theoretical_TT_unbinned_power_spec_D_ell(class_params)
    D_fac=ell*(ell+1)/(2*np.pi)
    Cltt_theory=Dltt_theory/D_fac
    compressed_data_dict={}
    for p in compression_and_binning_vectors:
        compressed_data_dict[p]=compression_and_binning_vectors[p].dot(Cltt_theory)

    return compressed_data_dict


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create compression vectors
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_compression_vectors_for_binned_data(theta_fid, params, with_low_ell):
    """
    Heavens et al MOPED weighting vectors
    theta_fid=dictionary of fiducial lambdaCDM parameters
    params=list of parameters to vary (must be a subset of parameters in theta_fid)
    returns compression vectors to be applied to planck binned power spectrum
    """
    class_param_dict=get_class_dict(theta_fid)
    #print(class_param_dict)
    dtheta_frac=get_dtheta_frac(theta_fid, params)
    #print('dtheta_frac', dtheta_frac)

    fisher=cmb_angular_power_spectrum.get_inverse_covmat_TT(with_low_ell)
    betas={}  #this will be the dictionary of weighting vectors
    dCdtheta={}
    for param in params:
        _, dCdtheta[param]=cmb_angular_power_spectrum.get_numerical_derivative_of_C_l_TT(class_param_dict, param, with_low_ell, frac=dtheta_frac[param])


    for i, param in enumerate(params):
        dCl=dCdtheta[param]

        num=fisher.dot(dCl)
        denom_sq=dCl.dot(fisher.dot(dCl))

        if i==0:
            betas[param]=num/np.sqrt(denom_sq)

        else:
            num_sum=0
            denom_sum=0
            for q in range(i):
                p_q=params[q]
                num_sum+=dCl.dot(betas[p_q])*betas[p_q]
                denom_sum+=(dCl.dot(betas[p_q]))**2

            betas[param]=(num-num_sum)/np.sqrt(denom_sq-denom_sum)
    return betas


def get_compression_vectors_for_unbinned_data(theta_fid, params, with_low_ell):
    '''
    returns compression vectors that both bin and compress theoretical CMB
    power spectra in one step
    '''
    binned_vecs=get_compression_vectors_for_binned_data(theta_fid, params, with_low_ell)
    unbinned_vecs={}
    for p in binned_vecs:
        unbinned_vecs[p]=cmb_angular_power_spectrum.include_binning_in_compression_vec(binned_vecs[p], with_low_ell)

    return unbinned_vecs

def get_compression_vectors_for_unbinned_data_from_binned(binned_vecs, with_low_ell):
    '''
    returns compression vectors that both bin and compress theoretical CMB
    power spectra in one step
    '''
    unbinned_vecs={}
    for p in binned_vecs:
        unbinned_vecs[p]=cmb_angular_power_spectrum.include_binning_in_compression_vec(binned_vecs[p], with_low_ell)
    return unbinned_vecs

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Extra helper functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_class_dict(theta_dict):
    CLASS_basic_dict={
            'output': 'tCl,pCl,lCl',
            'l_max_scalars': 3000,
            'lensing': 'yes',
            'N_ur':2.03066666667, #1 massive neutrino to match camb
            'N_ncdm': 1,
            'omega_ncdm' : 0.0006451439,
            'YHe':0.245341,
            'non linear' : 'halofit'}

    CLASS_basic_dict.update(theta_dict)

    return CLASS_basic_dict

def get_dtheta_frac(theta_dict, param_list):
    #~10% of 1 sigma
    errors_LCDM_planck2015={"h":0.0096, "omega_b":0.00023, "omega_cdm": 0.0022,
                "tau_reio": 0.019, "A_s": np.exp(0.036)/1e10, "n_s": 0.0062}
    dtheta_frac={}
    for p in param_list:
        try:
            dtheta_frac[p]=errors_LCDM_planck2015[p]/theta_dict[p]/10
        except:
            print('I do not have the error for this parameter:', p, 'using default')
            dtheta_frac[p]=0.01
    return dtheta_frac



if __name__=='__main__':
    ini_filename='../inifiles/LambdaCDM.ini'
    create_and_save_compression_data(ini_filename, with_low_ell=True)
    create_and_save_compression_data(ini_filename, with_low_ell=False)
