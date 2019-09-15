'''
CosMOPED example usage for Planck 2018 likelihoods
to use 2015 data, replace year=2018 with year=2015 when creating a PlanckLitePy object
'''
import numpy as np
from cosmoped_likelihood import CosMOPED
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
print('Testing CosMOPED for the LambdaCDM model using Planck 2018 data')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n')

print('Reading in theory power spectrum (generated by CLASS)\n')
# read some spectra to pass to the likelihood (can use CAMB/CLASS to generate these)
ls, Dltt, Dlte, Dlee = np.genfromtxt('Dl_planck2015fit.dat', unpack=True)
ellmin=int(ls[0])


# path to where compression vectors are stored, in this example for LambdaCDM parameters
path='compression_vectors/output/LambdaCDM/'

print('Test 1: high-l (l>=30) temperature only (TT)')
print('- - - - - - - - - - - - - - - - - - - - - -')
# create a CosMOPED object
TT2018=CosMOPED(path, year=2018, spectra='TT', use_low_ell_bins=False)
# call the likelihood function
loglike=TT2018.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
expected=-1.8016026266667178
print('CosMOPED log likelihood:', loglike)
print('Expected log likelihood:', expected)
print('Difference:', loglike-expected, '\n\n')

print('Test 2: high-l TT, TE and EE data + low-l TT bins')
print('- - - - - - - - - - - - - - - - - - - - - - - - - ')
# create a CosMOPED object
TTTEEE2018_lowTTbins=CosMOPED(path, year=2018, spectra='TTTEEE', use_low_ell_bins=True)

# call the likelihood function
loglike=TTTEEE2018_lowTTbins.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
expected=-1.9204015349888748
print('CosMOPED log likelihood:', loglike)
print('Expected log likelihood:', expected)
print('Difference:', loglike-expected, '\n\n')