# cosmoped
We apply the Massively Optimized Parameter Estimation and Data compression technique (MOPED, see Heavens, Jimenez & Lahav 2000 https://arxiv.org/abs/astro-ph/9911102) to the public Planck 2015 temperature likelihood, reducing the dimensions of the data space to one number per parameter of interest.

# required packages

In addition to numpy, scipy and matplotlib you will need 
* [CLASS](http://class-code.net/) and its [python wrapper](https://github.com/lesgourg/class_public/wiki/Python-wrapper)
* [emcee](https://emcee.readthedocs.io/en/latest/user/install/) if you want to use the sampling code (version 3 up recommended to use hdf5 backend)

# please cite

Planck 2015 likelihood paper (datafiles from the Planck plik-lite public likelihood code are used)
