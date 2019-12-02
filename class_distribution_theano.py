import numpy as np
import theano.tensor as tt
import theano


class GMM:

    def __init__(self, mu, isigma, sigma, weight, prior_mean, prior_cov):
        self.mu = mu
        self.isigma = isigma
        self.sigma = sigma
        self.weight = weight
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    def pdfunnormalized(self, x, beta):
        # d = len(x)
        mu = self.mu
        isigma = self.isigma
        weight = self.weight
        size = mu.shape
        if len(mu.shape) == 1:
            nmode = 1
        else:
            nmode = size[0]
        x0 = x - mu
        x1 = tt.reshape(x0, (nmode, 1, -1))
        x2 = tt.reshape(x0, (nmode, -1, 1))
        xsinv = tt.sum(tt.sum((x1 * x2) * isigma, axis=2), axis=1)
        pdf = tt.sum(weight * tt.exp(-0.5 * xsinv))**beta * self.multivariate_normal_pdf(x)**(1-beta)
        return pdf

    def derivativelogp(self, x, beta):
        theano.config.compute_test_value = 'ignore'
        y = tt.vector('y')
        logpdf_tt = theano.function([y], tt.grad(tt.log(self.pdfunnormalized(y, beta)), y))
        return logpdf_tt(x)

    def vderivativelogp(self, x, beta):
        theano.config.compute_test_value = 'ignore'
        y = tt.vector('y')
        logpdf_tt = theano.function([y], tt.grad(tt.log(self.pdfunnormalized(y, beta)), y))
        if len(x.shape) == 1:
            dlogp = logpdf_tt(x)
        else:
            dlogp = np.zeros(np.shape(x))
            for i in range(x.shape[0]):
                dlogp[i, :] = logpdf_tt(x[i, :])
        return dlogp

    def multivariate_normal_pdf(self, x):
        mean = self.prior_mean
        cov = self.prior_cov
        d = len(mean)
        x0 = x - mean
        xsinv = tt.dot(tt.dot(x0, np.linalg.inv(cov)), x0)
        pdf = tt.exp(-0.5 * xsinv) / tt.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return pdf















