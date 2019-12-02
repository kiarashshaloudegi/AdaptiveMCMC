import numpy as np
from scipy.stats import multivariate_normal
import pymc3 as pm
import time


class ClassMcmcSampler:

    def __init__(self, nchain, isigma, delta_mh, delta_mala):
        self.nchain = nchain
        self.isigma = isigma
        self.delta_mh = delta_mh
        self.delta_mala = delta_mala

    """takes sample"""
    def take_sample_mh(self, pdf, initial, nsample):
        nc = self.nchain  # number of chains
        # inverse_Sigma: standard deviation^2 or covariance matrix
        isigma = self.isigma
        delta = self.delta_mh
        ns = nsample.sum()
        idx = np.zeros(ns)

        for i in range(nc):
            if i == 0:
                idx[0:nsample[i]] = i
            else:
                idx[nsample[:i].sum():nsample[:i + 1].sum()] = i

        d = initial.shape[1]
        S = np.zeros((ns, d))  # X matrix in 2D
        prob_mass = np.zeros((ns, 1))

        for i in range(nc):

            def proposal(x, y):
                output_pro = np.exp(-0.5 * np.linalg.multi_dot(
                        [(x - y), isigma[i, :, :], (x - y)]))
                return output_pro

            def proprnd(x):
                output_rand = x + np.random.uniform(-1 * delta[i] / 2,
                                                    delta[i] / 2, (1, d))
                return output_rand

            if nsample[i] != 0:
                xs, prob = mh_sample(initial[i, :], pdf, nsample[i],
                                     proposal, proprnd)
                index = np.where(idx == i)
                S[index] = xs
                prob_mass[index, :] = prob
                initial[i, :] = xs[-1, :]
        s_idx = np.concatenate((S, idx.reshape(-1, 1)), axis=1)
        output = {'X': S, 'idx': idx, 'finalstate': initial,
                  'Prob_mass': np.concatenate(
                      (prob_mass, idx.reshape(prob_mass.shape)), axis=1),
                  'X_idx': s_idx}
        return output

    def take_sample_mala(self, pdf, dlogp, initial, nsample):
        nc = self.nchain  # number of chains
        delta = self.delta_mala
        ns = nsample.sum()
        idx = np.zeros(ns)

        for i in range(nc):
            if i == 0:
                idx[0:nsample[i]] = i
            else:
                idx[nsample[:i].sum():nsample[:i + 1].sum()] = i

        d = initial.shape[1]
        S = np.zeros((ns, d))  # X matrix in 2D
        prob_mass = np.zeros((ns, 1))

        for i in range(nc):

            if nsample[i] != 0:
                xs, prob = mala_sample(initial[i, :], pdf, dlogp, nsample[i], delta[i])
                index = np.where(idx == i)
                S[index] = xs
                prob_mass[index, :] = prob
                initial[i, :] = xs[-1, :]
        s_idx = np.concatenate((S, idx.reshape(-1,1)), axis=1)
        output = {'X': S, 'idx': idx, 'finalstate': initial,
                  'Prob_mass': np.concatenate(
                      (prob_mass, idx.reshape(prob_mass.shape)), axis=1),
                  'X_idx': s_idx}
        return output

    def take_sample_nuts(self, pdf, logpdf_pm, initial, nsample):
        nc = self.nchain  # number of chains
        ns = nsample.sum()
        idx = np.zeros(ns)

        for i in range(nc):
            if i == 0:
                idx[0:nsample[i]] = i
            else:
                idx[nsample[:i].sum():nsample[:i + 1].sum()] = i

        d = initial.shape[1]
        S = np.zeros((ns, d))  # X matrix in 2D
        prob_mass = np.zeros((ns, 1))

        for i in range(nc):
            s_time = time.time()
            with pm.Model() as model:
                pm.DensityDist('x', logpdf_pm, shape=d)
                step = pm.NUTS()
                trace = pm.sample(nsample[i], start={'x': initial[i]},
                                  step=step, njobs=1, chains=1,
                                  discard_tuned_samples=True,
                                  progressbar=False)
                xs = trace['x']
            print(time.time()-s_time)
            index = np.where(idx == i)
            S[index] = xs
        for i in range(ns):
            prob_mass[i] = pdf(S[i])
        output = {'X': S, 'idx': idx,
                  'Prob_mass': np.concatenate(
                      (prob_mass, idx.reshape(prob_mass.shape)), axis=1)}
        return output


def mh_sample(initial, pdf, num_sample, proposal, proprnd):

    def logpdf(x): return np.log(pdf(x))

    def logproposal(x, y): return np.log(proposal(x, y))

    dim = initial.shape[0]
    smpl = np.zeros((num_sample, dim))
    x0 = initial
    prob_mass = np.zeros((num_sample, 1))
    accept = np.zeros((num_sample, 1))
    u = np.log(np.random.uniform(0, 1, (num_sample, 1)))
    for i in range(num_sample):
        y0 = proprnd(x0).reshape(x0.shape)
        q1 = logproposal(x0, y0)
        q2 = logproposal(y0, x0)
        rho = (q1 + logpdf(y0)) - (q2 + logpdf(x0))
        rho_new = np.minimum(0, rho)

        if rho_new >= u[i]:
            x0 = y0
            accept[i] = 1

        smpl[i] = x0
        prob_mass[i, :] = pdf(x0)

    return smpl, prob_mass


def mala_sample(initial, pdf, dlogp, num_sample, h):
    """G. O. Roberts and J. S. Rosenthal (1998). "Optimal scaling of discrete approximations
    to Langevin diffusions".
    Journal of the Royal Statistical Society, Series B. 60 (1): 255–268.
    https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00123
    Also, inspired by the following code:
    https://michaellindon.github.io/lindonslog/mathematics/statistics/ mala-metropolis-adjusted-langevin-algorithm-julia/index.html
    """

    def logpdf(x): return np.log(pdf(x))

    def logproposal(x, y): return np.log(multivariate_normal.pdf(x.reshape(-1), (y+0.5*h*dlogp(y)).reshape(-1),
                                                                 h*np.eye(dim)))

    dim = initial.shape[0]
    smpl = np.zeros((num_sample, dim))
    x0 = initial
    prob_mass = np.zeros((num_sample, 1))
    accept = np.zeros((num_sample, 1))
    u = np.log(np.random.uniform(0, 1, (num_sample, 1)))
    for i in range(num_sample):
        y0 = np.random.multivariate_normal((x0 + 0.5*h*dlogp(x0)).reshape(-1), h*np.eye(dim)).reshape(x0.shape)
        q1 = logproposal(x0, y0)
        q2 = logproposal(y0, x0)
        rho = (q1 + logpdf(y0)) - (q2 + logpdf(x0))
        rho_new = np.minimum(0, rho)

        if rho_new >= u[i]:
            x0 = y0
            accept[i] = 1

        smpl[i] = x0
        prob_mass[i, :] = pdf(x0)

    return smpl, prob_mass


def kernel_coupling_mh_sample(initial, pdf, num_sample, nchain, delta):
    dim = initial.shape[1]

    def logpdf(x): return np.log(pdf(x))

    def proprnd(x):
        return x + np.random.uniform(-1 * delta / 2, delta / 2, (1, dim))

    def logproposal(x, y):
        delta_1 = y - x.reshape(1, -1)
        # z = np.sum(delta**2)
        return -np.sum(delta_1**2)

    dim = initial.shape[1]
    smpl = np.zeros((num_sample, dim))
    x0 = np.zeros(initial.shape)
    x0[:, :] = initial
    prob_mass = np.zeros((num_sample, 1))
    accept = np.zeros((num_sample, 1))
    u = np.log(np.random.uniform(0, 1, (num_sample, 1)))
    for i in range(num_sample):
        idx = np.random.randint(0, nchain, 1)
        y0 = np.zeros(x0.shape)
        y0[:, :] = x0
        y0[idx] = proprnd(x0[idx])
        q2 = logproposal(x0[idx], y0)
        q1 = logproposal(y0[idx], x0)
        rho = (q1 + logpdf(y0[idx])) - (q2 + logpdf(x0[idx]))
        rho_new = np.minimum(0, rho)

        if rho_new >= u[i]:
            x0[idx] = y0[idx]
            accept[i] = 1

        smpl[i] = x0[idx]
        # prob_mass[i, :] = pdf(x0)

    return smpl, prob_mass


def nuts_sample(initial, logpdf_pm, num_sample):
    d = len(initial)
    with pm.Model() as model:
        pm.DensityDist('x', logpdf_pm, shape=d)
        step = pm.NUTS()
        trace = pm.sample(num_sample, start={'x': initial},
                          step=step,
                          njobs=1,
                          chains=1, tune=125,
                          progressbar=False)
        return trace['x']

