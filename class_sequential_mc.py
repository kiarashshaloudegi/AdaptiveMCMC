import numpy as np
from classmcmcsampler import mh_sample, mala_sample


class SequentialMonteCarlo:
    def __init__(self, beta, nchain, method, param, dimension, number_step=1, cov_coeff=10):
        self.beta = np.sort(beta)
        self.nchain = nchain
        self.nstep = number_step
        self.method = method
        self.param = param
        self.dimension = dimension
        self.x0 = np.random.multivariate_normal(np.zeros(dimension),
                                                cov_coeff*np.eye(dimension), nchain)

    def _sampling_parallel_chains(self, x0, pdf, gradlogp, beta):
        s = np.zeros((self.nchain, 1, x0.shape[1]))
        p = np.zeros((self.nchain, 1, 1))

        def pdf_beta(x): return pdf(x, beta)

        def gradlogp_beta(x): return gradlogp(x, beta)

        for i in range(self.nchain):
            s[i], p[i] = self._mcmc_sample(x0[i], pdf_beta, gradlogp_beta, self.param)
        return s, p

    def _weight_cal(self, sample, pdf, beta1, beta2):
        tiny = 1e-30
        ns = self.nchain
        weight = np.zeros(ns)

        def pdf_beta1(x): return pdf(x, beta1)

        def pdf_beta2(x): return pdf(x, beta2)

        for i in range(ns):
            weight[i] = (pdf_beta2(sample[i])+tiny) / (pdf_beta1(sample[i])+tiny)
        weight = weight / weight.sum()
        return weight

    def sequential_mc(self, pdf, gradlogp):
        sample_final = np.zeros((len(self.beta), self.x0.shape[0], self.x0.shape[1]))
        sample = self.x0
        sample_final[0] = self.x0
        for epoch in range(1, len(self.beta)):
            weight = self._weight_cal(sample, pdf, self.beta[epoch-1], self.beta[epoch])
            sample = self._sample_generation(sample, weight)
            temp, _ = self._sampling_parallel_chains(sample, pdf, gradlogp, self.beta[epoch])

            sample_final[epoch] = temp.reshape(-1, self.x0.shape[1])
            sample = temp.reshape(-1, self.x0.shape[1])

        return sample_final

    def _mcmc_sample(self, x0, pdf, gradlogp, param):
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
        else:
            raise NotImplementedError('the {} sampler is not implemented'.format(self.method))

        return sample[-1], prob_mass[-1]

    def _sample_generation(self, sample, weight):
        ns = self.nchain
        n_new = np.floor(ns*weight)
        n_real = n_new.sum()
        n_residual = ns - n_real
        if n_residual != 0:
            distribution_multinomial = (ns*weight - n_new) / (ns - n_real)
            residual = np.random.multinomial(n_residual, distribution_multinomial, size=1)
        else:
            residual = np.zeros(n_new.shape)
        n_new = n_new + residual.reshape(-1)
        counter = 0
        newborn = np.zeros(sample.shape)
        for i in range(ns):
            if n_new[i] != 0:
                newborn[counter:counter+int(n_new[i]), :] = sample[i]
                counter += int(n_new[i])

        return sample


