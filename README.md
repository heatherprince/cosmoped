# CosMOPED

CosMOPED=Cosmological MOPED

We apply the Massively Optimized Parameter Estimation and Data compression technique (MOPED, see Heavens, Jimenez & Lahav 2000 https://arxiv.org/abs/astro-ph/9911102) to the public *Planck* 2015 temperature likelihood (https://arxiv.org/abs/1507.02704), and the *Planck* 2018 temperature and polarization likelihoods (https://arxiv.org/abs/1907.12875), reducing the dimensions of the data space to one number per parameter of interest.

# required packages

In addition to numpy, scipy and matplotlib you will need
* [CLASS](http://class-code.net/) and its [python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper)
* [emcee](https://emcee.readthedocs.io/en/latest/user/install/) if you want to use the sampling code (version 3 up recommended to use hdf5 backend)

# please cite

[*Planck* 2018 likelihood paper](https://arxiv.org/abs/1907.12875) or [*Planck* 2015 likelihood paper](https://www.aanda.org/articles/aa/abs/2016/10/aa26926-15/aa26926-15.html) ([arXiv version](https://arxiv.org/abs/1507.02704)) depending on which data you use, because we use datafiles from the *Planck* plik-lite public likelihood code

Our paper: arXiv link coming soon
