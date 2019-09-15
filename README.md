# CosMOPED

CosMOPED=Cosmological MOPED

We apply the Massively Optimized Parameter Estimation and Data compression technique (MOPED, see Heavens, Jimenez & Lahav 2000 https://arxiv.org/abs/astro-ph/9911102) to the public *Planck* 2015 temperature likelihood (https://arxiv.org/abs/1507.02704), and the *Planck* 2018 temperature and polarization likelihoods (https://arxiv.org/abs/1907.12875), reducing the dimensions of the data space to one number per parameter of interest.

# Required packages

To use the loglikelihood function:
* numpy
* scipy

Additional requirement for creating compression vectors:
* [CLASS](http://class-code.net/) and its [Python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper)

# Usage

## Likelihood
The CosMOPED compression vectors for the &Lambda;CDM parameters (h, omega_b, omega_cdm, tau_reio, A_s, n_s) already exist in compression_vectors/output, so to get the log likelihood for these you can don't need to make any new compression vectors.

NB: the naming conventions for parameters are the same as for the CLASS python wrapper (https://github.com/lesgourg/class_public/wiki/Python-wrapper), so omega_b = &Omega;<sub>b</sub>h<sup>2</sup> and omega_cdm = &Omega;<sub>CDM</sub>h<sup>2</sup>

```python
# import the CosMOPED class
from cosmoped_likelihood import CosMOPED

# initialize a CosMOPED object, specifying the path to the compression vectors (depends on model parameters)
# and which data you want to use (year, spectra and whether or not to use two low-ell temperature bins
TT2018=CosMOPED(path, year=2018, spectra='TT', use_low_ell_TT=False)

# call the likelihood function with your theoretical TT, TE, and EE  spectra (from e.g. CLASS or CAMB)
loglike=TT2018.loglike(Dltt, Dlte, Dlee, ellmin)
```

When initializing the CosMOPED object you can specify:
* path: to CosMOPED compression vectors for the parameters you are interested in
* year: 2015 or 2018 to use the *Planck* 2015 or 2018 data releases
* spectra: 'TT' for just temperature, or 'TTTEEE' for TT, TE and EE spectra
* use_low_ell: True to use two low-l temperature bins, False to use just l>=30 data

Notes on the CosMOPED log likelihood function:
* the log likelihood function expects the spectra in the form D<sub>l</sub>=l(l+1)/2&pi; C<sub>l</sub> 
* Dltt, Dlte and Dlee should all cover the same l range, usually from a minimum l value of 0 or 2
* ellmin=2 by default; if your spectra start at l=0 then specify this with ellmin=0

## Compression vectors






# Please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because we use datafiles from the *Planck* plik-lite public likelihood code

Our paper: arXiv link coming soon
