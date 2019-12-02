import numpy as np
from classmcmcsampler import mh_sample, mala_sample, nuts_sample


class ParallelTempering:
    def __init__(self, beta, number_step, dimension, method, param, cov_coeff=10):
        self.beta = np.sort(beta)
        self.nchain = len(beta)
        self.nstep = number_step
        self.method = method
        self.param = param
        self.x0 = np.random.multivariate_normal(np.zeros(dimension),
                                                cov_coeff * np.eye(dimension), self.nchain)

    def _sampling_parallel_chains(self, x0, pdf, gradlogp, logpdf_pm):
        s = np.zeros((self.nchain, self.nstep, x0.shape[1]))
        for i in range(self.nchain):
            def pdf_beta(x): return pdf(x, self.beta[i])

            def gradlogp_beta(x): return gradlogp(x, self.beta[i])

            def logpdf_pm_beta(x): return logpdf_pm(x, self.beta[i])

            s[i] = self._mcmc_sample(x0[i], pdf_beta, gradlogp_beta, self.param, logpdf_pm_beta)
        return s

    def parallel_tempreing(self, number_sample, pdf, gradlogp, logpdf_pm):

        max_epoch = int(number_sample/self.nstep)

        if number_sample % self.nstep != 0:
            raise NotImplementedError('number_sample has to be divisible by nstep')

        sample_final = np.zeros((self.nchain, self.nstep*max_epoch, self.x0.shape[1]))
        for epoch in range(max_epoch):
            temp = self._sampling_parallel_chains(self.x0, pdf, gradlogp, logpdf_pm)
            temp = self._swap_state(temp, pdf)
            sample_final[:, epoch*self.nstep:(epoch+1)*self.nstep, :] = temp

        return sample_final

    def _mcmc_sample(self, x0, pdf, gradlogp, param, logpdf_pm):
        d = len(x0)
        if self.method == 'RandomWalk':
            def proposal(x, y):
                output_pro = np.exp(-0.5 * np.linalg.multi_dot(
                    [(x - y), np.eye(d), (x - y)]))
                return output_pro

            def proprnd(x):
                output_rand = x + np.random.uniform(-1 * param / 2,
                                                    param / 2, (1, d))
                return output_rand
            sample, prob_mass = mh_sample(x0, pdf, self.nstep, proposal, proprnd)

        elif self.method == 'mala':
            sample, prob_mass = mala_sample(x0, pdf, gradlogp, self.nstep, param)
        elif self.method == 'nuts':
            sample = nuts_sample(x0, logpdf_pm, self.nstep)
        else:
            raise NotImplementedError('the {} sampler is not implemented'.format(self.method))

        return sample

    def _swap_state(self, sample, pdf):

        for i in range(self.nchain-1):
            def logpdf_beta1(x): return np.log(pdf(x, self.beta[i]))

            def logpdf_beta2(x): return np.log(pdf(x, self.beta[i+1]))

            # s1 = np.zeros(sample.shape[-1])
            # s2 = np.zeros(sample.shape[-1])
            s1 = sample[i, -1, :].copy()
            s2 = sample[i+1, -1, :].copy()

            prob_diff = (logpdf_beta1(s2) + logpdf_beta2(s1)
                         - logpdf_beta1(s1) - logpdf_beta2(s2))
            rho = np.minimum(0, prob_diff)

            if rho >= np.log(np.random.uniform(0, 1, 1)):
                sample[i, -1, :] = s2
                sample[i+1, -1, :] = s1

        return sample

