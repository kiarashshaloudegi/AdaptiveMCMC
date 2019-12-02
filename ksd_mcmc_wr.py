import os
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import sys
import time
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import matplotlib.pyplot as plt
from classmcmcsampler import ClassMcmcSampler
from classdistribution import GMM
from classdiscrepancy import StainMethod
from classbanditclustering import banditloss_sampleclustering, uniform_all
from class_parallel_tempering import ParallelTempering
from class_sequential_mc import SequentialMonteCarlo
from class_distribution import GMMBeta


def function_def(gmm, gmm_beta):
    constant = 1
    # free_param = 1
    kernel = StainMethod(free_param, constant, 0.5)

    def pdf(x_pdf): return gmm_beta.pdfunnormalized(x_pdf, beta=1)

    def gradlogp(x_grad): return gmm_beta.vderivativelogp(x_grad, beta=1)

    def pdf_beta(x_pdf, beta):
        return gmm_beta.pdfunnormalized(x_pdf, beta)

    def gradlogp_beta(x_grad, beta):
        return gmm_beta.derivativelogp(x_grad, beta)

    def loss(x, x_grad):
        return np.mean(kernel.iq_kernelp_matrix(x, x_grad))

    def logpdf_pm(x_pdf): return tt.log(gmm_beta.pdfunnormalized_theano(x_pdf, beta=1))

    def logpdf_pm_beta(x_pdf, beta): return tt.log(gmm_beta.pdfunnormalized_theano(x_pdf, beta))

    print('hi')
    x1 = np.linspace(-13, 13, 2000)
    y1 = np.linspace(-13, 13, 2000)
    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = np.zeros(X1.shape)
    for i in range(2000):
        for j in range(2000):
            Z1[i, j] = pdf(np.array([X1[i, j], Y1[i, j]]))

    # plt.figure()
    plt.contour(X1, Y1, Z1)
    # plt.show()
    fig = plt.gcf()
    plt.xticks([])
    plt.yticks([])
    fig.set_size_inches(22, 22 / 1.7)
    plt.savefig('contour_20_2.eps', format='eps', dpi=600)
    plt.show()

    return pdf, gradlogp, pdf_beta, gradlogp_beta, loss, logpdf_pm, kernel, logpdf_pm_beta


def nuts_sampling(s, x0, d, logpdf_pm, expected_value):

    with pm.Model() as model:
        pm.DensityDist('x', logpdf_pm, shape=d)
        step = pm.NUTS()
        trace = pm.sample(s, start={'x': x0[0]}, step=step,
                          njobs=1,
                          chains=1, tune=1000, discard_tuned_samples=False,
                          progressbar=False)
        trace1 = pm.sample(s, start={'x': x0[0]}, step=step,
                           njobs=1,
                           chains=1, tune=500, discard_tuned_samples=False,
                           progressbar=False)
    s_nuts = trace['x']
    error_nuts = np.linalg.norm(expected_value - s_nuts.mean(axis=0)) ** 2

    s_nuts1 = trace1['x']
    error_nuts1 = np.linalg.norm(expected_value - s_nuts1.mean(axis=0)) ** 2

    return error_nuts, error_nuts1


def parallel_samplers(sampler, pdf, gradlogp, logpdf_pm, nchain, d, x0):

    if method == 0:
        print('mh')
        output = sampler.take_sample_mh(
            pdf, x0, max_sample*np.ones(nchain, dtype=np.int))
    elif method == 1:
        print('mala')
        output = sampler.take_sample_mala(
            pdf, gradlogp, x0, max_sample*np.ones(nchain, dtype=np.int))
    elif method == 2:
        print('nuts')
        output = sampler.take_sample_nuts(
            pdf, logpdf_pm, x0, max_sample * np.ones(nchain, dtype=np.int))
    else:
        output = None

    iindex_all = output['idx']
    ssample_all = output['X']
    pprob_mass = output['Prob_mass'][:, 0]
    return iindex_all, ssample_all, pprob_mass


def tempering_based_samplers(s, x0, beta, pdf, gradlogp, expected_value, delta, d, logpdf_pm_beta):

    method_temp = 1
    if method_temp == 0:
        method_name = 'RandomWalk'
    elif method_temp == 1:
        method_name = 'mala'
    elif method_temp == 2:
        method_name = 'nuts'
    else:
        method_name = None

    if int(s/len(beta)) - s/len(beta) != 0:
        raise NotImplementedError('number of samples has to be divisible by number of chains')

    error_smc = np.zeros(len(delta))
    error_pt = np.zeros(len(delta))
    for i in range(len(delta)):
        if method_temp < 2:
            print(method_name)
            sampler = SequentialMonteCarlo(beta, int(s/len(beta)), method_name, delta[i], d)
            xs = sampler.sequential_mc(pdf, gradlogp)
            error_smc[i] = np.linalg.norm(expected_value - xs[-1].mean(axis=0)) ** 2

            sampler = ParallelTempering(beta, 25, d, method_name, delta[i])
            xs = sampler.parallel_tempreing(int(s / len(beta)), pdf, gradlogp, logpdf_pm_beta)
            error_pt[i] = np.linalg.norm(expected_value - xs[-1].mean(axis=0)) ** 2

        else:
            error_smc[i] = np.nan
            error_pt[i] = np.nan

    return error_smc, error_pt


def main():

    """initialization"""
    nmode = 20
    d_max = 24
    coefficient = np.linspace(1, 0.5, 6)
    # d = 2
    num_beta = 10
    # nchain = 100
    blk_size = [10, 100]
    np.random.seed(5*seed_mu)
    sigma = np.zeros((nmode, d, d))
    isigma = np.zeros((nmode, d, d))
    ss = 1 * np.random.uniform(0.5, 1, nmode)
    for i in range(nmode):
        sigma[i] = ss[i] * np.identity(d)
        isigma[i] = np.linalg.inv(sigma[i])
    mu = np.random.uniform(-10, 10, (nmode, d_max))
    mu = mu[:, :d]
    weight = np.random.uniform(0, 1, nmode)
    weight = weight / weight.sum()
    gmm = GMM(mu, isigma, sigma, weight)
    gmm_beta = GMMBeta(mu, isigma, sigma, weight, np.zeros(d), 10*np.eye(d))
    weight_matrix = np.zeros((nmode, 1))
    for i in range(nmode):
        weight_matrix[i] = weight[i] * np.sqrt(np.linalg.det(sigma[i]))
    expected_value = gmm.expectation()[0]
    sigma = np.zeros((nchain, d, d))
    isigma = np.zeros((nchain, d, d))
    for i in range(nchain):
        sigma[i] = np.identity(d)
        isigma[i] = np.linalg.inv(sigma[i])

    """sampler: param"""
    np.random.seed(1)
    delta = np.random.uniform(0.5, 5, nchain)
    assert delta.shape[0] == nchain, 'check the delta vector'
    sampler = ClassMcmcSampler(nchain, isigma, delta, delta)

    """beta"""
    beta_vec = np.sqrt(2)**np.arange(0, num_beta-1)/(np.sqrt(2)**(num_beta-2))
    beta_vec = np.concatenate([np.array([0]), beta_vec])

    """random seed"""
    np.random.seed(10000 * seed_initialization)

    """functions"""
    pdf, gradlogp, pdf_beta, gradlogp_beta, loss, logpdf_pm, kernel, logpdf_pm_beta = function_def(gmm, gmm_beta)

    """sampler"""
    x0_original = np.random.uniform(-10, 10, (nchain, d))
    x0 = np.zeros(x0_original.shape)
    x0[:, :] = x0_original
    s = time.time()
    iindex_all, ssample_all, pprob_mass = parallel_samplers(sampler, pdf, gradlogp, logpdf_pm, nchain, d, x0)
    print('sampling: {}'.format(time.time()-s))
    # print('hi')
    l_blk = len(blk_size)
    # error = np.zeros((l_blk, len(range(min_sample, max_sample, 2000)),
    #                   2*5 + 4 + nchain + 2*nchain))
    error = np.zeros((l_blk, len(range(min_sample, max_sample, 2000)),
                      2*5 + 4))
    for idx_s, s in enumerate(range(min_sample, max_sample, 2000)):
        sample_all = np.zeros((s * nchain, d))
        index_all = np.zeros(s * nchain)
        prob_mass = np.zeros(s * nchain)
        for i in range(nchain):
            sample_all[i*s:(i+1)*s] = ssample_all[i*max_sample:(i+1) * max_sample][:s]
            index_all[i*s:(i+1)*s] = iindex_all[i*max_sample:(i+1)*max_sample][:s]
            prob_mass[i*s:(i+1)*s] = pprob_mass[i*max_sample:(i+1)*max_sample][:s]
        #
        error_smc, error_pt = tempering_based_samplers(
            s, x0, beta_vec, pdf_beta, gradlogp_beta, expected_value, delta, d, logpdf_pm_beta)
        error_nuts, error_nuts1 = nuts_sampling(s, x0, d, logpdf_pm, expected_value)

        for b, blk in enumerate(blk_size):
            print(b, blk)
            knn = 6
            window = 5
            # alpha = 0.99
            error1 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'greedy',
                'entropy', knn, 'Kmean', alpha, 'all', window)
            print(error1)
            error2 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'greedy',
                'entropy', knn, 'DBSCAN', alpha, 'all', window)
            error3 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'greedy',
                'entropy', knn, 'GaussianMixture', alpha, 'all', window)
            #
            error5 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'ucb1',
                'entropy', knn, 'Kmean', alpha, 'all', window)
            print(error5)
            error6 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'ucb1',
                'entropy', knn, 'DBSCAN', alpha, 'all', window)
            error7 = banditloss_sampleclustering(
                sample_all, prob_mass, index_all, loss, gradlogp, s, blk,
                nchain, d, expected_value, 'uniform', 'ucb1',
                'entropy', knn, 'GaussianMixture', alpha, 'all', window)
            #
            echain, euni, _, wuni1 = uniform_all(
                sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, 'non-bbox', 'std', knn, kernel,
                'Kmean', alpha)
            _, _, _, wuni2 = uniform_all(
                sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, 'non-bbox', 'entropy', knn, kernel,
                'Kmean', alpha)
            print(euni, wuni1, wuni2)
            _, _, _, wuni3 = uniform_all(
                sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, 'non-bbox', 'entropy', knn, kernel,
                'DBSCAN', alpha)
            _, _, _, wuni4 = uniform_all(
                sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, 'non-bbox', 'entropy', knn, kernel,
                'GaussianMixture', alpha)
            errorbbox = np.nan
            temp = np.concatenate(
                (error1, error2, error3, error5, error6, error7, np.array(
                    [errorbbox, euni, wuni1, wuni2, wuni3, wuni4, error_nuts, error_nuts1]),
                 echain, error_smc, error_pt)
            )

            error[b, idx_s] = temp
        if s <= 5000:
            _, _, errorbbox, _ = uniform_all(
                sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, 'bbox', 'entropy', 6, kernel,
                'Kmean', 0.99)
            error[:, idx_s, 6*5] = errorbbox
    return error


if __name__ == '__main__':
    '''this code has two seeds'''
    seed = int(sys.argv[1])
    idx_free = 2
    free_vec = [0.001, 0.01, 1, 10, 100]
    free_param = free_vec[idx_free]
    seed_mu = seed // 500
    d = 2
    seed_initialization = seed % 50
    nchain = 10 + 10 * (seed % 500 // 50)
    method = 2
    max_sample = 10000 + 500
    min_sample = 20000
    alpha = 0.99
    start_time = time.time()
    error_all = main()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    name = 'ksd_clustering_ratio_mu_{}_chain_{}_seed{}.npz'.format(seed_mu, nchain, seed_initialization)
    np.savez(name, error=error_all)
