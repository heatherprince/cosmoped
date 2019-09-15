# CosMOPED

CosMOPED=Cosmological MOPED

We apply the Massively Optimized Parameter Estimation and Data compression technique (MOPED, see Heavens, Jimenez & Lahav 2000 https://arxiv.org/abs/astro-ph/9911102) to the public *Planck* 2015 temperature likelihood (https://arxiv.org/abs/1507.02704), and the *Planck* 2018 temperature and polarization likelihoods (https://arxiv.org/abs/1907.12875), reducing the dimensions of the data space to one number per parameter of interest.

# required packages

To use the loglikelihood function:
* numpy
* scipy

Additional requirement for creating compression vectors:
* [CLASS](http://class-code.net/) and its [Python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper)

# usage

The CosMOPED compression vectors for the &Lambda;CDM parameters (h, omega_b, omega_cdm, tau_reio, A_s, n_s) already exist in compression_vectors/output, so to get the log likelihood for these you can don't need to make any new compression vectors.

NB: the naming conventions for parameters are the same as for the CLASS python wrapper (https://github.com/lesgourg/class_public/wiki/Python-wrapper), so omega_b = &Omega;<sub>b</sub> h<sup>2</sup> and omega_cdm = &Omega;<sub>CDM</sub> h<sup>2</sup>

# please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because we use datafiles from the *Planck* plik-lite public likelihood code

Our paper: arXiv link coming soon
