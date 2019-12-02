import sys
import os
address = os.getcwd() + '/py_ite'
sys.path.insert(0, address)
import numpy as np
import time
import cvxpy as cvx
from cvxpy import *
from collections import deque
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import kmeans, whiten, kmeans2
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
import ite
from ite.cost.x_factory import co_factory


def blackbox_iportance_weight(Q):
    Q = (Q + Q.T) / 2
    n = Q.shape[0]
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.quad_form(x, Q))
    constraints = [0 <= x, x <= 1, cvx.sum_entries(x) == 1]
    prob = cvx.Problem(objective, constraints)
    try:
        result = prob.solve()
    except SolverError:
        result = prob.solve(solver=SCS)
    # result = prob.solve()
    return np.array(x.value)


def set_cover(universe, subsets):
    """
    Find a family of subsets that covers the universal set
    available online:
        http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
    """
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def weight_cal_entropy(x, p, index, cover, method, alpha):
    x_org = x
    k_knn = 60
    cost_name = 'BHRenyi_KnnK'  # dim >= 1
    co = co_factory(cost_name, mult=True, alpha=alpha, k=k_knn)
    k = len(list(cover))
    w = np.zeros(k)
    h = np.zeros(k)
    s = np.zeros(k)
    d = x.shape[1]

    x_min = np.min(x)
    xs = x - x_min
    scale = np.max(xs)
    xs = xs / scale
    # xs = x
    c_mean = np.zeros((k, d))
    for i in range(k):
        idx = np.where(np.isin(index, list(cover[i])))
        w[i] = np.log(np.mean(p[idx]**(alpha-1)))
        s[i] = np.sqrt(np.linalg.det(np.cov(x[idx].T)))
        c_mean[i] = np.mean(x_org[idx], axis=0)
        if xs[idx].shape[0] >= (k_knn+1):
            h[i] = co.estimation(xs[idx])
        elif 2 <= xs[idx].shape[0] < (k_knn+1):
            co = co_factory(cost_name, mult=True, alpha=alpha,
                            k=xs[idx].shape[0]-1)
            h[i] = co.estimation(xs[idx])
            co = co_factory(cost_name, mult=True, alpha=alpha,
                            k=k_knn)

    temp = (h - w / (1 - alpha))
    weight_m = np.exp(temp)
    weight_e = weight_m / np.nansum(weight_m)
    return weight_e, weight_e, c_mean


def weight_cal(x, p, index, cover, method):
    k = len(list(cover))
    w = np.zeros(k)
    s = np.zeros(k)
    d = x.shape[1]
    c_mean = np.zeros((k, d))
    for i in range(k):
        idx = np.where(np.isin(index, list(cover[i])))
        w[i] = np.mean(p[idx])
        s[i] = np.sqrt(np.linalg.det(np.cov(x[idx].T)))
        # s[i] = np.sqrt(np.diag(np.cov(x[idx].T)).sum())
        c_mean[i] = np.mean(x[idx], axis=0)

    weight = w * s
    weight_m = weight
    weight_e = weight_m / weight_m.sum()
    return weight, weight_e, c_mean


def epsilon_greedy(c, mean, eps):
    idx = np.array(list(c))
    r = np.random.rand(1)
    if r <= (1 - eps):
        k = np.argmin(abs(mean[idx]))
        k = idx[k]
    else:
        k = np.random.choice(idx)
    return k


def ucb1(c, mean, t, t_arm):
    idx = np.array(list(c))
    n_arm = mean.shape[0]
    cfd = 2 * np.log(t) * np.ones(n_arm)
    cfd = np.sqrt(cfd / t_arm)
    ucb = mean - cfd
    ucb = ucb[idx]
    k = np.argmin(ucb)
    k = idx[k]
    return k


def banditloss_sampleclustering(sample_all, prob_mass, index_all,
                                loss, gradlogp, s, blk, nchain, d,
                                expected_value, method1, method2, method3,
                                n_neighbors, c_method, alpha, method4, window):
    sss = time.time()
    t_total = int(s / blk)
    t_arm = np.ones(nchain)
    sample = np.zeros((s, d))
    p = np.zeros(s)
    s_idx = 10000 * np.ones(s)
    id_xnn = np.zeros((nchain, nchain))
    id_set = []
    arm_mean = np.zeros(nchain)
    mean_short = []
    for i in range(nchain):
        idx = np.where(index_all == i)
        sample[i * blk:(i + 1) * blk] = sample_all[idx][:blk]
        p[i * blk:(i + 1) * blk] = prob_mass[idx][:blk]
        s_idx[i * blk:(i + 1) * blk] = index_all[idx][:blk]
        arm_mean[i] = loss(sample_all[idx][:blk],
                           gradlogp(sample_all[idx][:blk]))
        temp = arm_mean[i]
        mean_short.append(deque([temp], window))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='ball_tree'
                            ).fit(sample[:nchain * blk])
    _, indices = nbrs.kneighbors(sample[:nchain * blk])
    for i in range(nchain):
        indices_k_unique = np.unique(indices[i * blk:(i + 1) * blk])
        idices_k_neigh = np.unique(s_idx[:nchain * blk][indices_k_unique]
                                   ).astype(int)
        id_xnn[idices_k_neigh, i] = 1
        id_xnn[i, idices_k_neigh] = 1
        id_set.append(set(idices_k_neigh))

    time_range = np.arange(nchain, t_total)
    xn = sample[:nchain * blk]  # contains the last batch ...
    # of samples from each sampler
    weight = 1/nchain * np.ones(nchain)
    for t in time_range:
        universe = set(range(nchain))
        subsets = id_set
        cover = set_cover(universe, subsets)

        if method1 == 'normal_std':
            _, weight, _ = weight_cal(sample, p, s_idx, cover, method2)
        elif method1 == 'normal_entropy':
            _, weight, _ = weight_cal_entropy(sample, p, s_idx, cover,
                                              method2, alpha)
        elif method1 == 'uniform':
            weight = np.ones(len(cover)) / len(cover)

        c = np.random.choice(len(cover), p=weight)
        if method4 == 'window':
            arm_mean_bandit = np.zeros(nchain)
            for i in range(nchain):
                arm_mean_bandit[i] = np.array(mean_short[i]).mean()
        elif method4 == 'all':
            arm_mean_bandit = arm_mean

        if method2 == 'greedy':
            k = epsilon_greedy(cover[c], arm_mean_bandit, 0.1)
        elif method2 == 'ucb1':
            k = ucb1(cover[c], arm_mean_bandit, t, t_arm)
        t_arm[k] += 1
        tk = int(t_arm[k])
        idx = np.where(index_all == k)
        sample[t * blk:(t + 1) * blk] = sample_all[idx][
                                        (tk - 1) * blk:tk * blk]
        p[t * blk:(t + 1) * blk] = prob_mass[idx][(tk - 1) * blk:tk * blk]
        s_idx[t * blk:(t + 1) * blk] = k
        mean_temp = loss(sample[t * blk:(t + 1) * blk],
                         gradlogp(sample[t * blk:(t + 1) * blk]))
        mean_short[k].append(mean_temp)
        arm_mean[k] = (1 - 1 / tk) * arm_mean[k] + mean_temp / tk
        xn[k * blk:(k + 1) * blk] = sample[t * blk:(t + 1) * blk]

        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm='ball_tree'
                                ).fit(xn)
        _, indices = nbrs.kneighbors(xn)
        id_xnn = np.zeros((nchain, nchain))
        # id_set = []
        for i in range(nchain):
            indices_k_unique = np.unique(indices[i * blk:(i + 1) * blk])
            idices_k_neigh = np.unique(
                s_idx[:nchain * blk][indices_k_unique]
            ).astype(int)
            id_xnn[idices_k_neigh, i] = 1
            id_xnn[i, idices_k_neigh] = 1
            id_set.append(set(idices_k_neigh))
    # print('bandit: {}'.format(time.time()-sss))
    error1 = np.linalg.norm(expected_value - sample.mean(axis=0)) ** 2

    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='ball_tree'
                            ).fit(sample)
    _, indices = nbrs.kneighbors(sample)
    id_xnn = np.zeros((nchain, nchain))
    id_set = []
    for i in range(nchain):
        indices_k_unique = np.unique(indices[i * blk:(i + 1) * blk])
        idices_k_neigh = np.unique(s_idx[indices_k_unique]
                                   ).astype(int)
        id_xnn[idices_k_neigh, i] = 1
        id_xnn[i, idices_k_neigh] = 1
        id_set.append(set(idices_k_neigh))
    universe = set(range(nchain))
    subsets = id_set
    cover = set_cover(universe, subsets)
    _, weight, mean = weight_cal(sample, p, s_idx, cover, 'w')
    mean = np.sum(mean * np.repeat(weight.reshape(-1, 1), d, axis=1),
                  axis=0)
    error2 = np.linalg.norm(expected_value - mean) ** 2

    l_cover = len(cover)
    for i in range(l_cover):
        for j in range(l_cover):
            if i != j:
                temp1 = cover[i]
                temp2 = cover[j]
                temp = temp1 - (temp1 - temp2)
                if temp:
                    cover[i] = temp1 - temp
                    cover[j] = temp2 - temp
                    cover.append(temp)

    _, weight, mean = weight_cal(sample, p, s_idx, cover, 'w')
    mean = np.sum(mean * np.repeat(weight.reshape(-1, 1), d, axis=1),
                  axis=0)
    error3 = np.linalg.norm(expected_value - mean) ** 2

    _, weight, mean = weight_cal_entropy(sample, p, s_idx, cover, 'w', alpha)
    mean = np.nansum(mean * np.repeat(weight.reshape(-1, 1), d, axis=1),
                     axis=0)
    error4 = np.linalg.norm(expected_value - mean) ** 2

    ss = time.time()
    whitened = whiten(sample)
    if c_method == 'kmeans2':
        _, label = kmeans2(whitened, nchain)
    elif c_method == 'Kmean':
        label = KMeans(n_clusters=nchain,
                       random_state=2000).fit_predict(sample)
    elif c_method == 'DBSCAN':
        label = DBSCAN(eps=3, min_samples=10).fit_predict(sample)
    elif c_method == 'SpectralClustering':
        label = SpectralClustering(n_clusters=nchain).fit_predict(sample)
    elif c_method == 'GaussianMixture':
        label = GaussianMixture(n_components=nchain).fit_predict(sample)

    cover = []
    unique = np.unique(label)
    for i in unique:
        if np.sum(np.where(label == i)):
            cover.append(set([i]))
    nchain = len(cover)
    mean = np.zeros((nchain, d))
    for j, i in enumerate(unique):
        idx = label == i
        mean[j] = sample[idx].mean(axis=0)

    if method3 == 'std':
        _, weight, _ = weight_cal(sample, p, label, cover, 'w')
    elif method3 == 'entropy':
        # start_time_entropy = time.time()
        _, weight, _ = weight_cal_entropy(sample, p, label, cover, 'w', alpha)
        # elapsed_time_entropy = time.time() - start_time_entropy
        # print('entropy time:', elapsed_time_entropy)

    fmean = np.zeros(d)
    for i, _ in enumerate(unique):
            fmean = fmean + weight[i] * mean[i]

    error5 = np.linalg.norm(expected_value - fmean) ** 2
    # print('re-weight: {}'.format(time.time()-ss))
    # error5 = np.nan

    return np.nan, np.nan, np.nan, np.nan, error5


def uniform_all(sample_all, prob_mass, index_all, loss, gradlogp, s,
                nchain, d, expected_value, method1, method2, n_neighbors,
                kernel, c_method, alpha):
    length = int(s/nchain)
    sample = np.zeros((int(s), d))
    p = np.zeros(int(s))
    s_idx = 10000 * np.ones(s)
    mean = np.zeros((nchain, d))
    error = np.zeros(nchain)
    for i in range(nchain):
        idx = np.where(index_all == i)
        sample[i*length:(i+1)*length] = sample_all[idx][:length]
        p[i*length:(i+1)*length] = prob_mass[idx][:length]
        s_idx[i*length:(i+1)*length] = i
        error[i] = np.linalg.norm(expected_value -
                                  sample_all[idx].mean(axis=0)) ** 2
    error1 = np.linalg.norm(expected_value - sample.mean(axis=0)) ** 2

    if method1 == 'bbox':
        s_grad = gradlogp(sample)
        if s <= 6000:
            kp = kernel.iq_kernelp_matrix(sample, s_grad)
            bb_w = blackbox_iportance_weight(kp)  # black-box importance weight
        else:
            bb_w = np.ones((s, 1)) * np.nan
        mean = np.sum(np.repeat(bb_w.reshape(-1, 1), d, axis=1) * sample,
                      axis=0)
        error2 = np.linalg.norm(expected_value - mean) ** 2
    else:
        error2 = np.nan

    whitened = whiten(sample)
    if c_method == 'kmeans2':
        _, label = kmeans2(whitened, nchain)
    elif c_method == 'Kmean':
        label = KMeans(n_clusters=nchain,
                       random_state=2000).fit_predict(sample)
    elif c_method == 'DBSCAN':
        label = DBSCAN(eps=3, min_samples=10).fit_predict(sample)
    elif c_method == 'SpectralClustering':
        label = SpectralClustering(n_clusters=nchain).fit_predict(sample)
    elif c_method == 'GaussianMixture':
        label = GaussianMixture(n_components=nchain).fit_predict(sample)

    cover = []
    unique = np.unique(label)
    for i in unique:
        if np.sum(np.where(label == i)):
            cover.append(set([i]))
    nchain = len(cover)
    mean = np.zeros((nchain, d))
    for j, i in enumerate(unique):
        idx = label == i
        mean[j] = sample[idx].mean(axis=0)

    if method2 == 'std':
        _, weight, _ = weight_cal(sample, p, label, cover, 'w')
    elif method2 == 'entropy':
        _, weight, _ = weight_cal_entropy(sample, p, label, cover, method2,
                                          alpha)
    fmean = np.zeros(d)
    for i, _ in enumerate(unique):
            fmean = fmean + weight[i] * mean[i]

    error3 = np.linalg.norm(expected_value - fmean) ** 2
    return error, error1, error2, error3
