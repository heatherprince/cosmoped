import os
import time
import numpy as np

from classy import Class
import emcee
print(emcee.__version__)

import read_inifiles
import cosmoped_likelihood


class SampleCosMOPED():
    def __init__(self, sampling_inifile):
        ini_dict=read_inifiles.read_sampling_inifile(sampling_inifile)

        # directory to save chains - - - - - - - - - - - - - - - - - - - - - -
        self.save_dir = ini_dict['save_dir']
        if self.save_dir[-1]!='/': # deal with whether it has last /
            self.save_dir+='/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_file_prefix = ini_dict['save_file_prefix']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # things needed for emcee sampler object - - - - - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        nwalkers = int(ini_dict['nwalkers'])

        # read compression vectors
        compression_data_folder = ini_dict['compression_vector_folder']
        if compression_data_folder[-1]!='/': # deal with whether it has last /
            compression_data_folder+='/'
        with_low_ell = bool(ini_dict['with_low_ell'])
        compression_vectors, compressed_data = read_inifiles.read_compression_vectors_and_compressed_data(
                        compression_data_folder, with_low_ell)

        parameter_file = ini_dict['parameter_file']
        fid_param_vals, start_vals, prior_bounds = read_inifiles.read_values(parameter_file)

        prior_file = ini_dict['prior_file']
        prior_gaussian = read_inifiles.read_gaussian_priors(prior_file)

        # check that the compression datafiles match each other and params you want to sample
        assert set(compression_vectors.keys()) == set(compressed_data.keys())
        assert set(compression_vectors.keys()) == set(start_vals.keys())
        assert set(prior_gaussian.keys()).issubset(set(start_vals.keys()))

        # default to match Planck unless overwritten by inifile
        class_basic_dict={
                'output': 'tCl,pCl,lCl',
                'l_max_scalars': 3000,
                'lensing': 'yes',
                'non linear' : 'halofit',
                'N_ur':2.03066666667,
                'N_ncdm': 1,
                'omega_ncdm' : 0.0006451439,
                'YHe':0.245341 }
        class_basic_dict.update(fid_param_vals)

        params_to_sample=list(start_vals.keys())
        ndim=len(params_to_sample)
        logprob_fn = cosmoped_likelihood.logprob

        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = self.save_dir+self.save_file_prefix+'.h5'
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

        class_obj = Class()
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn,
            args=(params_to_sample, prior_bounds, prior_gaussian, class_obj,
                  class_basic_dict, compression_vectors, compressed_data), backend=backend)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # things for running emcee - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # create starting ball of walkers
        pos0=list(start_vals.values())
        size=[p/1e6 for p in pos0]
        self.pos=emcee.utils.sample_ball(pos0, size, size=nwalkers)

        # total number of steps
        self.saveas = self.save_dir+self.save_file_prefix+'.dat'
        self.burnin = int(ini_dict['burnin'])
        self.nsteps = int(ini_dict['nsteps'])
        self.nsteps_check_autocorr = int(ini_dict['nsteps_check_autocorr'])

        # to test timing of logprob:
        self.pos0=list(start_vals.values())
        self.args=(params_to_sample, prior_bounds, prior_gaussian,
              class_obj, class_basic_dict, compression_vectors, compressed_data)
        self.logprob_fn=logprob_fn

        class_obj.struct_cleanup()
        class_obj.empty()

    def sample(self):
        pos=self.pos
        # autocorrelation stuff from https://emcee.readthedocs.io/en/latest/tutorials/monitor/

        # This will be useful to testing convergence
        old_tau = np.inf

        nloops=int(np.ceil(self.nsteps/self.nsteps_check_autocorr))

        # We'll track how the average autocorrelation time estimate changes
        autocorr = np.empty(nloops)

        for i in range(0, nloops):
            start = time.time()
            pos, prob, state = self.sampler.run_mcmc(pos, self.nsteps_check_autocorr, store=True) #try with backend if this works
            end = time.time()
            print('emcee sampling took ', (end-start), 'seconds for ', self.nsteps_check_autocorr, ' steps in loop ', i)
            np.savetxt(self.save_dir+'flatchain.dat', self.sampler.flatchain)
            np.savetxt(self.save_dir+'lnprob.dat', self.sampler.flatlnprobability)

            tau = self.sampler.get_autocorr_time(tol=0)
            autocorr[i] = np.mean(tau)
            f=open(self.save_dir+'autocorrelation.dat','ab')
            np.savetxt(f, np.array([[(i+1)*self.nsteps_check_autocorr, autocorr[i]]]))
            f.close()

            converged = np.all(tau * self.nsteps_check_autocorr < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau



        # for sample in self.sampler.sample(self.pos, iterations=max_n, store=True, progress=True): #giving memory issues?
        #     # Only check convergence every 100 steps
        #     if self.sampler.iteration % self.nsteps_check_autocorr:
        #         continue
        #     # Compute the autocorrelation time so far
        #     # Using tol=0 means that we'll always get an estimate even
        #     # if it isn't trustworthy
        #     np.savetxt(self.save_dir+'flatchain.dat', self.sampler.flatchain)
        #     np.savetxt(self.save_dir+'lnprob.dat', self.sampler.flatlnprobability)
        #
        #     tau = self.sampler.get_autocorr_time(tol=0)
        #     autocorr[index] = np.mean(tau)
        #     index += 1
        #     # Check convergence
        #     converged = np.all(tau * self.nsteps_check_autocorr < self.sampler.iteration)
        #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        #     if converged:
        #         break
        #     old_tau = tau

        n = self.nsteps_check_autocorr*np.arange(1, i+1)
        acor = autocorr[:i+1]

        np.savetxt(self.save_dir+'autocorrelation_end.dat',+np.column_stack((n, acor)))


    def hdf5_to_textfile(self):
        filename = self.save_dir+self.save_file_prefix+'.h5'
        reader = emcee.backends.HDFBackend(filename)
        # tau = reader.get_autocorr_time()
        # burnin = int(2*np.max(tau))
        # thin = int(0.5*np.min(tau))

        samples = reader.get_chain(discard=self.burnin, flat=True)#, thin=thin)
        log_prob_samples = reader.get_log_prob(discard=self.burnin)#, flat=True, thin=thin)

        import IPython; IPython.embed()
        np.savetxt(self.save_dir+self.save_file_prefix+'_flatchain.dat', samples)
        np.savetxt(self.save_dir+self.save_file_prefix+'_flatlnprob.dat', samples)

    def time_likelihood(self):
        start = time.time()
        lnprob=self.logprob_fn(self.pos0, *(self.args))
        end=time.time()
        print('total time for one likelihood call=', end-start)

    def sample_incremental_save_emceev2(self):
        pos=self.pos
        # if self.burnin>0:
        #     print('starting ', self.burnin, ' burnin steps')
        #     start = time.time()
        #     pos, prob, state  = self.sampler.run_mcmc(pos, self.burnin)
        #     self.sampler.reset()
        #     end=time.time()
        #     print('burnin time: %f' %(end-start))

        print('starting mcmc sampling')
        nloops=2
        nsteps_incremental_write=int(np.ceil(self.nsteps/nloops))
        for i in range(0, nloops):
            start = time.time()
            pos, prob, state=self.sampler.run_mcmc(pos, nsteps_incremental_write)
            end = time.time()
            print('emcee sampling took ', (end-start), 'seconds for ', self.nsteps, ' steps in loop ', i)
            # save
            f=open(self.save_dir+'flatchain.dat','ab')
            np.savetxt(f, self.sampler.flatchain)
            f.close()
            f=open(self.save_dir+'lnprob.dat','ab')
            np.savetxt(f, self.sampler.flatlnprobability)
            f.close()
            np.savetxt(self.save_dir+'pos_after_'+str((i+1)*nsteps)+'_steps.dat', pos)

        tau = sampler.get_autocorr_time(tol=0)
        np.savetxt(self.savedir+'autocorr.dat', tau)



if __name__=='__main__':
    # read command line argument for inifile name
    # initialise everything
    # sample

    sampling_object=SampleCosMOPED('../inifiles/sample_compressed_LCDM.ini')
    sampling_object.time_likelihood()
    sampling_object.sample()
    #sampling_object.hdf5_to_textfile()
