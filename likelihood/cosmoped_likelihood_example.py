import numpy as np
from cosmoped_likelihood import CosMOPED

# path to where compression vectors are stored, in this example for LambdaCDM parameters
path='../compression_vectors/output/testing/'

# create a CosMOPED object
TTTEEE2018_lowTTbins=CosMOPED(path, year=2018, spectra='TTTEEE', use_low_ell_TT=True)

# read some spectra to pass to the likelihood (can use CAMB/CLASS to generate these)
ls, Dltt, Dlte, Dlee = np.genfromtxt('data/Dl.dat', unpack=True)
ellmin=int(ls[0])

# call the likelihood function
loglike=TTTEEE2018_lowTTbins.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
print('CosMOPED likelihood (2018 high-l TT, TE, EE + low-l TT bins):', loglike)

# suppose we only want the high-ell temperature
TT2015=CosMOPED(path, year=2015, spectra='TT', use_low_ell_TT=False)
loglike=TT2015.loglike(Dltt, Dlte, Dlee, ellmin) #ellmin = 2 by default
print('CosMOPED likelihood (2015 high-l TT):', loglike)
