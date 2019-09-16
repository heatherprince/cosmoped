# CosMOPED: a compressed Planck likelihood

CosMOPED=Cosmological MOPED

To compute the likelihood for the LambdaCDM model using CosMOPED you only need 6 compression vectors (one for each parameter) and 6 numbers (from compressing the *Planck* data using the 6 compression vectors). Using these, the likelihood of a theory power spectrum given the *Planck* data is just the product of 6 one-dimensional Gaussians. Extended cosmological models just require computing extra compression vectors. For more details on how this works see https://arxiv.org/abs/1909.05869

We apply the Massively Optimized Parameter Estimation and Data compression technique (MOPED, see [Heavens, Jimenez & Lahav, 2000](https://arxiv.org/abs/astro-ph/9911102)) to the public [*Planck* 2015](https://arxiv.org/abs/1507.02704) temperature likelihood, and the [*Planck* 2018](https://arxiv.org/abs/1907.12875) temperature and polarization likelihoods, reducing the dimensions of the data space to one number per parameter of interest.



# Required packages

To use the log likelihood function:
* numpy
* scipy

Additional requirement for creating compression vectors:
* [CLASS](http://class-code.net/) and its [Python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper) to compute the theory power spectrum

# Usage

For a quick start example see cosmoped_likelihood_example.py

## Compression vectors
The Lambda CDM compression vectors are pre-computed so if that's what you want then skip straight to the likelihood section.

If you want to create compression vectors for a different cosmological model you can do this by running

```bash
python compression_vectors.py inifiles/settings.ini
```

where the settings.ini inifile points to the appropriate compression_inifile which specifies which parameters to calculate compression vectors for and what their fiducial values should be.

NB: the naming conventions for parameters in the compression inifile are the same as for the [CLASS python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper), so omega_b = &Omega;<sub>b</sub>h<sup>2</sup> and omega_cdm = &Omega;<sub>CDM</sub>h<sup>2</sup> etc

## Likelihood
1. import the CosMOPED class
```python
from cosmoped_likelihood import CosMOPED
```

2. initialize a CosMOPED object, specifying
  * path: to CosMOPED compression vectors for the parameters you are interested in
  * year: 2015 or 2018 to use the *Planck* 2015 or 2018 data releases
  * spectra: 'TT' for just temperature, or 'TTTEEE' for TT, TE and EE spectra
  * use_low_ell: True to use two low-l temperature bins, False to use just l>=30 data
```python
path='../compression_vectors/output/LambdaCDM/'
TT2018_LambdaCDM=CosMOPED(path, year=2018, spectra='TT', use_low_ell_bins=False)
```

A note on compression vectors:
* The CosMOPED compression vectors for the &Lambda;CDM parameters (h, omega_b, omega_cdm, tau_reio, A_s, n_s) already exist in compression_vectors/output, so to get the log likelihood for these you don't need to make any new compression vectors.
* note omega_b = &Omega;<sub>b</sub>h<sup>2</sup> and omega_cdm = &Omega;<sub>CDM</sub>h<sup>2</sup> (CLASS python wrapper naming conventions)



3. call the likelihood function with your theoretical TT, TE, and EE  spectra (from e.g. CLASS or CAMB)
```python
loglike=TT2018_LambdaCDM.loglike(Dltt, Dlte, Dlee, ellmin)
```
Note:
* the log likelihood function expects the spectra in the form D<sub>l</sub>=l(l+1)/2&pi; C<sub>l</sub> 
* Dltt, Dlte and Dlee should all cover the same l range, usually from a minimum l value of 0 or 2
* ellmin=2 by default; if your spectra start at l=0 then specify this with ellmin=0




# Please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because we use datafiles from the *Planck* plik-lite public likelihood code

Our paper: https://arxiv.org/abs/1909.05869
